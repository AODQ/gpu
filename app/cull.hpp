#include "gfx.hpp"

namespace Cull {

	void init(
		gfx::Device const & context,
		VkCommandPool const & commandPool
	);
	void shutdown(gfx::Device const & context);

	void update(
		gfx::Device const & context,
		VkCommandBuffer commandBuffer,
		f32m44 const & viewProj
	);

	void draw(
		gfx::Device const & context,
		VkCommandBuffer commandBuffer,
		f32m44 const & viewProj
	);

	void resolveDepth(
		gfx::Device const & context,
		VkCommandBuffer commandBuffer,
		VkImage const depthImage,
		VkImageView const depthImageView
	);

	u32 lastVisibleCount();
	u32 totalInstanceCount();

	VkImageView imageHiz();
	void imageHizTransition(VkCommandBuffer const & commandBuffer);
}
