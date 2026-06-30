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
	auto readF32v4 = [](rapidjson::Value const & v, char const * key, f32v4 & out) {
		if (!v.HasMember(key) || !v[key].IsArray()) { return; }
		auto const arr = v[key].GetArray();
		if (arr.Size() < 4u) { return; }
		out.x = arr[0u].GetFloat();
		out.y = arr[1u].GetFloat();
		out.z = arr[2u].GetFloat();
		out.w = arr[3u].GetFloat();
	};
	auto readF32 = [](rapidjson::Value const & v, char const * key, f32 & out) {
		if (!v.HasMember(key) || !v[key].IsNumber()) { return; }
		out = v[key].GetFloat();
	};
	auto readU32 = [](rapidjson::Value const & v, char const * key, u32 & out) {
		if (!v.HasMember(key) || !v[key].IsUint()) { return; }
		out = v[key].GetUint();
	};
	auto readMaterial = [&](rapidjson::Value const & obj, SceneMaterial & sm) {
		if (obj.HasMember("name") && obj["name"].IsString()) {
			sm.name = obj["name"].GetString();
		}
		GpuMorMaterial & p = sm.params;
		readF32v4(obj, "baseColor", p.baseColor);
		readF32(obj, "baseMetalness", p.baseMetalness);
		readF32(obj, "specularRoughness", p.specularRoughness);
		readF32v3(obj, "emissiveColor", p.emissiveColor);
		readF32(obj, "emissiveLuminance", p.emissiveLuminance);
		readF32v3(obj, "specularColor", p.specularColor);
		readF32(obj, "specularWeight", p.specularWeight);
		readF32(obj, "specularIor", p.specularIor);
		readF32(obj, "coatWeight", p.coatWeight);
		readF32(obj, "coatRoughness", p.coatRoughness);
		readF32v3(obj, "fuzzColor", p.fuzzColor);
		readF32(obj, "fuzzWeight", p.fuzzWeight);
		readF32(obj, "fuzzRoughness", p.fuzzRoughness);
		readF32(obj, "subsurfaceWeight", p.subsurfaceWeight);
		readF32v3(obj, "subsurfaceColor", p.subsurfaceColor);
		readF32(obj, "subsurfaceRadius", p.subsurfaceRadius);
		readF32(obj, "transmissionWeight", p.transmissionWeight);
		readF32v3(obj, "transmissionColor", p.transmissionColor);
		readF32(obj, "transmissionDepth", p.transmissionDepth);
		readF32(obj, "thinFilmWeight", p.thinFilmWeight);
		readF32(obj, "thinFilmIor", p.thinFilmIor);
		readF32(obj, "thinFilmThickness", p.thinFilmThickness);
		readF32(obj, "thinFilmThicknessMin", p.thinFilmThicknessMin);
		readF32(obj, "specularRoughnessAnisotropy", p.specularRoughnessAnisotropy);
		readF32(obj, "specularAnisotropyRotation", p.specularAnisotropyRotation);
		readF32(obj, "transmissionDispersionAbbeNumber", p.transmissionDispersionAbbeNumber);
		readF32(obj, "alphaCutoff", p.alphaCutoff);
		readF32(obj, "geometryOpacity", p.geometryOpacity);
		readU32(obj, "flags", p.flags);
	};

	for (auto const & e : doc["entities"].GetArray()) {
		if (!e.IsObject() || !e.HasMember("type") || !e["type"].IsString()) { continue; }
		char const * const type = e["type"].GetString();
		f32v3 position = {};
		readF32v3(e, "position", position);

		if (strcmp(type, "instance") == 0) {
			EntityInstance inst = { .filename = {}, .rotation = {}, .scale = 1.0f, .materials = {} };
			if (e.HasMember("filename") && e["filename"].IsString()) {
				inst.filename = e["filename"].GetString();
			}
			readF32v3(e, "rotation", inst.rotation);
			if (e.HasMember("scale") && e["scale"].IsNumber()) {
				inst.scale = e["scale"].GetFloat();
			}
			if (e.HasMember("materials") && e["materials"].IsArray()) {
				for (auto const & matObj : e["materials"].GetArray()) {
					if (!matObj.IsObject()) { continue; }
					SceneMaterial sm {};
					readMaterial(matObj, sm);
					inst.materials.push_back(std::move(sm));
				}
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
	auto writeF32v4 = [&](char const * key, f32v4 const & v) {
		writer.Key(key);
		writer.StartArray();
		writer.Double((double)v.x);
		writer.Double((double)v.y);
		writer.Double((double)v.z);
		writer.Double((double)v.w);
		writer.EndArray();
	};
	auto writeMaterial = [&](SceneMaterial const & sm) {
		GpuMorMaterial const & p = sm.params;
		writer.StartObject();
		writer.Key("name"); writer.String(sm.name.c_str());
		writeF32v4("baseColor", p.baseColor);
		writer.Key("baseMetalness"); writer.Double((double)p.baseMetalness);
		writer.Key("specularRoughness"); writer.Double((double)p.specularRoughness);
		writeF32v3("emissiveColor", p.emissiveColor);
		writer.Key("emissiveLuminance"); writer.Double((double)p.emissiveLuminance);
		writeF32v3("specularColor", p.specularColor);
		writer.Key("specularWeight"); writer.Double((double)p.specularWeight);
		writer.Key("specularIor"); writer.Double((double)p.specularIor);
		writer.Key("coatWeight"); writer.Double((double)p.coatWeight);
		writer.Key("coatRoughness"); writer.Double((double)p.coatRoughness);
		writeF32v3("fuzzColor", p.fuzzColor);
		writer.Key("fuzzWeight"); writer.Double((double)p.fuzzWeight);
		writer.Key("fuzzRoughness"); writer.Double((double)p.fuzzRoughness);
		writer.Key("subsurfaceWeight"); writer.Double((double)p.subsurfaceWeight);
		writeF32v3("subsurfaceColor", p.subsurfaceColor);
		writer.Key("subsurfaceRadius"); writer.Double((double)p.subsurfaceRadius);
		writer.Key("transmissionWeight"); writer.Double((double)p.transmissionWeight);
		writeF32v3("transmissionColor", p.transmissionColor);
		writer.Key("transmissionDepth"); writer.Double((double)p.transmissionDepth);
		writer.Key("thinFilmWeight"); writer.Double((double)p.thinFilmWeight);
		writer.Key("thinFilmIor"); writer.Double((double)p.thinFilmIor);
		writer.Key("thinFilmThickness"); writer.Double((double)p.thinFilmThickness);
		writer.Key("thinFilmThicknessMin"); writer.Double((double)p.thinFilmThicknessMin);
		writer.Key("specularRoughnessAnisotropy"); writer.Double((double)p.specularRoughnessAnisotropy);
		writer.Key("specularAnisotropyRotation"); writer.Double((double)p.specularAnisotropyRotation);
		writer.Key("transmissionDispersionAbbeNumber"); writer.Double((double)p.transmissionDispersionAbbeNumber);
		writer.Key("alphaCutoff"); writer.Double((double)p.alphaCutoff);
		writer.Key("geometryOpacity"); writer.Double((double)p.geometryOpacity);
		writer.Key("flags"); writer.Uint(p.flags);
		writer.EndObject();
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
					writer.Key("materials");
					writer.StartArray();
					for (SceneMaterial const & sm : d.materials) {
						writeMaterial(sm);
					}
					writer.EndArray();
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
