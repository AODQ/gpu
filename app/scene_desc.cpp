#include "scene_desc.hpp"

#include <imgui.h>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>

#include <cstdio>

SceneDesc scene_desc_load(std::filesystem::path const & path)
{
	SceneDesc desc;
	FILE * f = fopen(path.c_str(), "r");
	if (!f) { return desc; }

	char buf[65536];
	rapidjson::FileReadStream stream(f, buf, sizeof(buf));
	rapidjson::Document doc;
	doc.ParseStream(stream);
	fclose(f);

	if (doc.HasParseError() || !doc.IsObject()) { return desc; }
	if (!doc.HasMember("instances") || !doc["instances"].IsArray()) { return desc; }

	for (auto const & inst : doc["instances"].GetArray()) {
		if (!inst.IsObject()) { continue; }

		SceneInstance si;
		si.scale = 1.0f;

		if (inst.HasMember("filename") && inst["filename"].IsString()) {
			si.filename = inst["filename"].GetString();
		}

		auto readVec = [&](char const * key, f32v3 & out) {
			if (!inst.HasMember(key) || !inst[key].IsArray()) { return; }
			auto const arr = inst[key].GetArray();
			if (arr.Size() < 3u) { return; }
			out.x = arr[0u].GetFloat();
			out.y = arr[1u].GetFloat();
			out.z = arr[2u].GetFloat();
		};
		readVec("position", si.position);
		readVec("rotation", si.rotation);
		if (inst.HasMember("scale") && inst["scale"].IsNumber()) {
			si.scale = inst["scale"].GetFloat();
		}

		desc.instances.push_back(std::move(si));
	}

	if (doc.HasMember("lights") && doc["lights"].IsArray()) {
		for (auto const & l : doc["lights"].GetArray()) {
			if (!l.IsObject()) { continue; }
			GpuLight light = { .position = {}, .radius = 10.0f, .color = {}, };
			auto readVec = [&](char const * key, f32v3 & out) {
				if (!l.HasMember(key) || !l[key].IsArray()) { return; }
				auto const arr = l[key].GetArray();
				if (arr.Size() < 3u) { return; }
				out.x = arr[0u].GetFloat();
				out.y = arr[1u].GetFloat();
				out.z = arr[2u].GetFloat();
			};
			readVec("position", light.position);
			readVec("color", light.color);
			if (l.HasMember("radius") && l["radius"].IsNumber()) {
				light.radius = l["radius"].GetFloat();
			}
			desc.lights.push_back(light);
		}
	}

	if (doc.HasMember("ddgi_volumes") && doc["ddgi_volumes"].IsArray()) {
		for (auto const & v : doc["ddgi_volumes"].GetArray()) {
			if (!v.IsObject()) { continue; }
			SceneDescDdgiVolume vol = {
				.origin = {},
				.probeSpacing = { 1.0f, 1.0f, 1.0f },
				.probeCounts = { 8u, 4u, 8u },
				.raysPerProbe = 128u,
			};
			auto readF32v3 = [&](char const * key, f32v3 & out) {
				if (!v.HasMember(key) || !v[key].IsArray()) { return; }
				auto const arr = v[key].GetArray();
				if (arr.Size() < 3u) { return; }
				out.x = arr[0u].GetFloat();
				out.y = arr[1u].GetFloat();
				out.z = arr[2u].GetFloat();
			};
			auto readU32v3 = [&](char const * key, u32v3 & out) {
				if (!v.HasMember(key) || !v[key].IsArray()) { return; }
				auto const arr = v[key].GetArray();
				if (arr.Size() < 3u) { return; }
				out.x = arr[0u].GetUint();
				out.y = arr[1u].GetUint();
				out.z = arr[2u].GetUint();
			};
			readF32v3("origin", vol.origin);
			readF32v3("probe_spacing", vol.probeSpacing);
			readU32v3("probe_counts", vol.probeCounts);
			desc.ddgiVolumes.push_back(vol);
		}
	}

	return desc;
}

void scene_desc_save(SceneDesc const & desc, std::filesystem::path const & path)
{
	FILE * f = fopen(path.c_str(), "w");
	if (!f) { return; }

	char buf[65536];
	rapidjson::FileWriteStream stream(f, buf, sizeof(buf));
	rapidjson::PrettyWriter<rapidjson::FileWriteStream> writer(stream);

	writer.StartObject();
	writer.Key("instances");
	writer.StartArray();
	for (SceneInstance const & inst : desc.instances) {
		writer.StartObject();

		writer.Key("filename");
		writer.String(inst.filename.c_str());

		auto writeVec = [&](char const * key, f32v3 const & v) {
			writer.Key(key);
			writer.StartArray();
			writer.Double((double)v.x);
			writer.Double((double)v.y);
			writer.Double((double)v.z);
			writer.EndArray();
		};
		writeVec("position", inst.position);
		writeVec("rotation", inst.rotation);
		writer.Key("scale");
		writer.Double((double)inst.scale);

		writer.EndObject();
	}
	writer.EndArray();

	writer.Key("lights");
	writer.StartArray();
	for (GpuLight const & l : desc.lights) {
		writer.StartObject();
		auto writeVec = [&](char const * key, f32v3 const & v) {
			writer.Key(key);
			writer.StartArray();
			writer.Double((double)v.x);
			writer.Double((double)v.y);
			writer.Double((double)v.z);
			writer.EndArray();
		};
		writeVec("position", l.position);
		writeVec("color", l.color);
		writer.Key("radius");
		writer.Double((double)l.radius);
		writer.EndObject();
	}
	writer.EndArray();

	writer.Key("ddgi_volumes");
	writer.StartArray();
	for (SceneDescDdgiVolume const & vol : desc.ddgiVolumes) {
		writer.StartObject();
		auto writeF32v3 = [&](char const * key, f32v3 const & v) {
			writer.Key(key);
			writer.StartArray();
			writer.Double((double)v.x);
			writer.Double((double)v.y);
			writer.Double((double)v.z);
			writer.EndArray();
		};
		auto writeU32v3 = [&](char const * key, u32v3 const & v) {
			writer.Key(key);
			writer.StartArray();
			writer.Uint(v.x);
			writer.Uint(v.y);
			writer.Uint(v.z);
			writer.EndArray();
		};
		writeF32v3("origin", vol.origin);
		writeF32v3("probe_spacing", vol.probeSpacing);
		writeU32v3("probe_counts", vol.probeCounts);
		writer.EndObject();
	}
	writer.EndArray();

	writer.EndObject();
	fclose(f);
}

void scene_desc_add_instance(
	SceneDesc & desc,
	std::string filename,
	f32v3 const position
) {
	desc.instances.emplace_back(SceneInstance {
		.filename = std::move(filename),
		.position = position,
		.rotation = {},
		.scale = 1.0f,
	});
}

i32 scene_desc_imgui(
	SceneDesc & desc,
	std::filesystem::path & savePath
) {
	ImGui::Begin("scene object list");

	if (savePath.empty()) {
		static char sSaveAsInput[512] = {};
		ImGui::SetNextItemWidth(200.0f);
		ImGui::InputText("##saveas", sSaveAsInput, sizeof(sSaveAsInput));
		ImGui::SameLine();
		if (ImGui::Button("Save As") && sSaveAsInput[0] != '\0') {
			savePath = sSaveAsInput;
			scene_desc_save(desc, savePath);
		}
	} else {
		if (ImGui::Button("Save")) {
			scene_desc_save(desc, savePath);
		}
		ImGui::SameLine();
		if (ImGui::Button("Load")) {
			desc = scene_desc_load(savePath);
		}
	}

	ImGui::Separator();

	i32 selectedIdx = -1;
	for (i32 i = 0; i < (i32)desc.instances.size(); ++i) {
		ImGui::PushID(i);
		std::string const label = (
			std::filesystem::path(desc.instances[i].filename).stem().string()
		);
		if (ImGui::Button(label.c_str())) {
			selectedIdx = i;
		}
		ImGui::PopID();
	}

	ImGui::End();
	return selectedIdx;
}
