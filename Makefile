.PHONY: fmt docs-regen docs-check

fmt:
	cargo fmt
	ruff format
	find inference-* -type f \( -name "*.metal" -o -name "*.c" -o -name "*.cu" -o -name "*.hpp" -o -name "*.h" -o -name "*.cpp" \) -exec clang-format -i {} +
docs-regen:
	cargo test -p inference-cli regenerate_cli_reference -- --ignored
	cargo test -p inference-server-core regenerate_openapi -- --ignored
	cargo test -p inference-core regenerate_supported_models -- --ignored
	python3 docs/scripts/render_pyi.py
	python3 docs/scripts/render_examples.py

docs-check:
	cargo test -p inference-cli docgen::cli_reference_matches_committed
	cargo test -p inference-server-core openapi_matches_committed
	cargo test -p inference-core supported_models_matches_committed
	python3 docs/scripts/render_pyi.py --check
