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
	writer.EndObject();
	fclose(f);
}

void scene_desc_add_instance(SceneDesc & desc, std::string filename)
{
	desc.instances.push_back({
		.filename = std::move(filename),
		.position = {},
		.rotation = {},
		.scale = 1.0f,
	});
}

i32 scene_desc_imgui(SceneDesc & desc, std::filesystem::path & savePath)
{
	ImGui::Begin("Scene");

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

	i32 removeIdx = -1;
	i32 focusIdx = -1;
	for (i32 i = 0; i < (i32)desc.instances.size(); ++i) {
		ImGui::PushID(i);
		if (ImGui::CollapsingHeader(desc.instances[i].filename.c_str())) {
			SceneInstance & inst = desc.instances[i];
			if (ImGui::SmallButton("focus")) { focusIdx = i; }
			ImGui::SameLine();
			if (ImGui::SmallButton("remove")) { removeIdx = i; }
			ImGui::DragFloat3("position", &inst.position.x, 0.01f);
			ImGui::DragFloat3("rotation", &inst.rotation.x, 0.5f);
			ImGui::DragFloat("scale", &inst.scale, 0.01f, 0.001f, 1000.0f);
		}
		ImGui::PopID();
	}
	if (removeIdx >= 0) {
		desc.instances.erase(desc.instances.begin() + removeIdx);
	}

	ImGui::End();
	return focusIdx;
}
