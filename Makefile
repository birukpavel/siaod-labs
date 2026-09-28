VARIANT := 2
PYTHON  := python3

# Репозиторий курса (в нём scripts/generate_data.py)
FOS ?= ../data-structures-and-algorithms
SET ?= arrays

.PHONY: help data lab01 lint

help:
	@echo "Вариант $(VARIANT), seed $$(( 30 + $(VARIANT) ))"
	@echo "  make data FOS=<путь> [SET=arrays|ops]  — данные варианта"
	@echo "  make lab01                             — ЛР 1"
	@echo "  make lint                              — black и pylint"

data:
	$(PYTHON) $(FOS)/scripts/generate_data.py \
		--variant $(VARIANT) --only $(SET) --out data/generated

lab01:
	$(PYTHON) lab01/lab01_complexity.py --variant $(VARIANT) --out lab01/figures

lint:
	$(PYTHON) -m black --check .
	$(PYTHON) -m pylint $(wildcard lab*/*.py)
