.PHONY: fmt docs-regen docs-check

fmt:
	cargo fmt
	ruff format
	find crates -type f \( -name "*.metal" -o -name "*.c" -o -name "*.cu" -o -name "*.hpp" -o -name "*.h" -o -name "*.cpp" \) -exec clang-format -i {} +
docs-regen:
	cargo test -p inference-cli regenerate_cli_reference -- --ignored
	cargo test -p inference-server-core regenerate_openapi -- --ignored
	python3 bindings/python/scripts/generate_types.py
	python3 bindings/scripts/generate_native.py
	cargo test -p inference-selection regenerate_supported_models -- --ignored
	python3 docs/scripts/render_python_api.py
	python3 docs/scripts/render_examples.py

docs-check:
	cargo test -p inference-cli docgen::cli_reference_matches_committed
	cargo test -p inference-server-core openapi_matches_committed
	cargo test -p inference-selection supported_models_matches_committed
	python3 docs/scripts/render_python_api.py --check
	python3 bindings/scripts/generate_native.py --check
