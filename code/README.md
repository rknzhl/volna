# Код

Папка [notebooks/](../notebooks) в корне репозитория содержит ноутбуки 01–07.

`waves/` содержит генераторы физических полей, загрузчики The Well и NOAA,
модели прогноза и математические проверки. Данные хранятся отдельно в `data/`.

`lag_ssa/` содержит исходники исследовательских методов и экспериментов.
Результаты записываются в `results/lag_ssa`; общая настройка пути находится
в `lag_ssa/project_paths.py`.

| Направление | Основные файлы | Пример |
|---|---|---|
| Взвешенный MSSA | `method.py`, `nodal_spatial.py`, `mssa_column_encoding.py` | `run_controlled.py`, `run_actual_wave_mssa_baseline.py` |
| Геометрия и TDA | `geometric_phase_frontend.py`, `geometric_phase_encoding.py`, `cohomology_primitivity.py` | [ноутбук 06](../notebooks/06_phase_ssa.ipynb) |
| Перенос между окнами: прежний исследовательский прототип | `tangent_overlap.py`, `tangent_overlap_angles_v2.py` | `run_tangent_overlap_pilot.py` |
| Спектральные подходы | `ssa_character.py`, `phase_lattice.py`, `factor_ssa.py` | `run_character_ssa.py`, `run_factor_ssa.py` |

[Ноутбук 07](../notebooks/07_tangent_overlap.ipynb) содержит самодостаточный
точный двухволновой пример: касательный перенос, интегрирование периодов и фаз,
тор и восстановление потоками полей.
Общая математическая конструкция описана в [PDF](../research/phase_ssa_method.pdf).

Файлы методов находятся в одной папке, поскольку используют общие компоненты.
Префиксы обозначают назначение: `run_` запускает расчёт, `audit_` и `check_`
проверяют результат, `plot_` и `draw_` строят рисунки.

Пример запуска из корня:

```bash
.venv/bin/python code/lag_ssa/run_tangent_overlap_pilot.py --help
jupyter lab notebooks
```
