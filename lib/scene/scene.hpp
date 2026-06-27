#pragma once

#include <srat/core-math.hpp>

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace scene
{

struct EntityInstance
{
	std::string filename;
	f32v3 rotation;
	f32 scale;
};

struct EntityLight
{
	f32v3 color;
	f32 radius;
};

using EntityData = std::variant<EntityInstance, EntityLight>;

struct Entity
{
	u32 id;
	f32v3 position;
	EntityData data;
};

struct Desc
{
	std::vector<Entity> entities;
	u32 nextEntityId = 0u;
};

[[nodiscard]] Desc desc_load(std::filesystem::path const & path);
void desc_save(Desc const & desc, std::filesystem::path const & path);
void desc_add_entity(Desc & desc, EntityData data, f32v3 position = {});

[[nodiscard]] f32m44 entity_model_matrix(Entity const & entity, EntityInstance const & inst);
[[nodiscard]] f32v3 entity_focus_target(Entity const & entity);
[[nodiscard]] std::string entity_label(Entity const & entity, u32 nthOfType);

} // namespace scene
