// unit-test/main.cpp
// doctest entry point: parse --windowed, init vkof, run tests, shutdown.

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <vkof/vkof.hpp>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
	// Separate our --windowed flag from the doctest flags
	bool windowed = false;
	std::vector<char *> filtered;
	filtered.push_back(argv[0]);
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--windowed") == 0) {
			windowed = true;
		} else {
			filtered.push_back(argv[i]);
		}
	}
	int filteredArgc = static_cast<int>(filtered.size());

	if (windowed) {
		vkof::init();
	} else {
		vkof::init_headless();
	}

	doctest::Context ctx(filteredArgc, filtered.data());

	// Exclude [windowed] test-suite when running headless
	if (!windowed) {
		ctx.addFilter("test-suite-exclude", "[windowed]");
	}

	int const result = ctx.run();

	vkof::shutdown();
	return result;
}
