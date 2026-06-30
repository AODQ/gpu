#pragma once
#include <vkof/vkof.hpp>

struct MaterialTableImages {
	vkof::Image kullaContyEnergyImage;
	vkof::Image kullaContyEnergyAvgImage;
	vkof::Image zeltnerLtcParamImage;
	vkof::Sampler sampler;
	u32 kullaContyEnergyHandle;
	u32 kullaContyEnergyAvgHandle;
	u32 zeltnerLtcParamHandle;
};

[[nodiscard]] MaterialTableImages material_tables_create();
void material_tables_destroy(MaterialTableImages & tables);
