#include <scene/scene.hpp>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>

#include <cstdio>
#include <cstring>

scene::Desc scene::desc_load(std::filesystem::path const & path)
{
	Desc desc;
	FILE * f = fopen(path.c_str(), "r");
	if (!f) { return desc; }

	char buf[65536];
	rapidjson::FileReadStream stream(f, buf, sizeof(buf));
	rapidjson::Document doc;
	doc.ParseStream(stream);
	fclose(f);

	if (doc.HasParseError() || !doc.IsObject()) { return desc; }
	if (!doc.HasMember("entities") || !doc["entities"].IsArray()) { return desc; }

	auto readF32v3 = [](rapidjson::Value const & v, char const * key, f32v3 & out) {
		if (!v.HasMember(key) || !v[key].IsArray()) { return; }
		auto const arr = v[key].GetArray();
		if (arr.Size() < 3u) { return; }
		out.x = arr[0u].GetFloat();
		out.y = arr[1u].GetFloat();
		out.z = arr[2u].GetFloat();
	};

	for (auto const & e : doc["entities"].GetArray()) {
		if (!e.IsObject() || !e.HasMember("type") || !e["type"].IsString()) { continue; }
		char const * const type = e["type"].GetString();
		f32v3 position = {};
		readF32v3(e, "position", position);

		if (strcmp(type, "instance") == 0) {
			EntityInstance inst = { .filename = {}, .rotation = {}, .scale = 1.0f };
			if (e.HasMember("filename") && e["filename"].IsString()) {
				inst.filename = e["filename"].GetString();
			}
			readF32v3(e, "rotation", inst.rotation);
			if (e.HasMember("scale") && e["scale"].IsNumber()) {
				inst.scale = e["scale"].GetFloat();
			}
			desc_add_entity(desc, std::move(inst), position);
		} else if (strcmp(type, "light") == 0) {
			EntityLight light = { .color = {}, .radius = 10.0f };
			readF32v3(e, "color", light.color);
			if (e.HasMember("radius") && e["radius"].IsNumber()) {
				light.radius = e["radius"].GetFloat();
			}
			desc_add_entity(desc, light, position);
		}
	}

	return desc;
}

void scene::desc_save(Desc const & desc, std::filesystem::path const & path)
{
	FILE * f = fopen(path.c_str(), "w");
	if (!f) { return; }

	char buf[65536];
	rapidjson::FileWriteStream stream(f, buf, sizeof(buf));
	rapidjson::PrettyWriter<rapidjson::FileWriteStream> writer(stream);

	auto writeF32v3 = [&](char const * key, f32v3 const & v) {
		writer.Key(key);
		writer.StartArray();
		writer.Double((double)v.x);
		writer.Double((double)v.y);
		writer.Double((double)v.z);
		writer.EndArray();
	};

	writer.StartObject();
	writer.Key("entities");
	writer.StartArray();

	for (Entity const & entity : desc.entities) {
		writer.StartObject();
		std::visit(
			[&](auto const & d) {
				using T = std::decay_t<decltype(d)>;
				if constexpr (std::is_same_v<T, EntityInstance>) {
					writer.Key("type"); writer.String("instance");
					writeF32v3("position", entity.position);
					writer.Key("filename"); writer.String(d.filename.c_str());
					writeF32v3("rotation", d.rotation);
					writer.Key("scale"); writer.Double((double)d.scale);
				} else if constexpr (std::is_same_v<T, EntityLight>) {
					writer.Key("type"); writer.String("light");
					writeF32v3("position", entity.position);
					writeF32v3("color", d.color);
					writer.Key("radius"); writer.Double((double)d.radius);
				}
			},
			entity.data
		);
		writer.EndObject();
	}

	writer.EndArray();
	writer.EndObject();
	stream.Flush();
	fclose(f);
}

void scene::desc_add_entity(Desc & desc, EntityData data, f32v3 const position)
{
	desc.entities.emplace_back(Entity {
		.id = desc.nextEntityId++,
		.position = position,
		.data = std::move(data),
	});
}

f32m44 scene::entity_model_matrix(Entity const & entity, EntityInstance const & inst)
{
	static constexpr f32 kDegToRad = 0.017453292519943295f;
	return (
		f32m44_translate(entity.position.x, entity.position.y, entity.position.z)
		* f32m44_rotate_y(inst.rotation.y * kDegToRad)
		* f32m44_rotate_x(inst.rotation.x * kDegToRad)
		* f32m44_rotate_z(inst.rotation.z * kDegToRad)
		* f32m44_scale(inst.scale, inst.scale, inst.scale)
	);
}

f32v3 scene::entity_focus_target(Entity const & entity)
{
	return entity.position;
}

std::string scene::entity_label(Entity const & entity, u32 const nthOfType)
{
	if (auto const * ei = std::get_if<EntityInstance>(&entity.data)) {
		return std::filesystem::path(ei->filename).stem().string();
	}
	if (std::holds_alternative<EntityLight>(entity.data)) {
		return "light " + std::to_string(nthOfType);
	}
	return "entity " + std::to_string(nthOfType);
}
