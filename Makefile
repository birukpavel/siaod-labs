VARIANT := 2
PYTHON  := python3

# Репозиторий курса (в нём scripts/generate_data.py)
FOS ?= ../data-structures-and-algorithms
SET ?= arrays

.PHONY: help data lint

help:
	@echo "Вариант $(VARIANT), seed $$(( 30 + $(VARIANT) ))"
	@echo "  make data FOS=<путь> [SET=arrays|ops]  — данные варианта"
	@echo "  make lint                              — black и pylint"

data:
	$(PYTHON) $(FOS)/scripts/generate_data.py \
		--variant $(VARIANT) --only $(SET) --out data/generated

lint:
	$(PYTHON) -m black --check .
	$(PYTHON) -m pylint $(wildcard lab*/*.py)
