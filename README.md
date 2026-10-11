# Пространственно-временные ряды и SSA

## Основные файлы

| Файл | Содержание |
|---|---|
| [Описание метода — PDF](research/phase_ssa_method.pdf) · [LaTeX](research/phase_ssa_method.tex) | Восстановление фаз и показаний через перенос между окнами. |
| [Ноутбук 06](notebooks/06_phase_ssa.ipynb) | Поиск циклов и построение углов по показаниям датчиков. |
| [Ноутбук 07](notebooks/07_tangent_overlap.ipynb) | Пример из PDF: две распространяющиеся волны, касательный перенос и обратное восстановление. |
| [Спектральный конспект](research/interpretation.pdf) | Связь SSA/MSSA, гармоник и фазового пространства. |

[Все ноутбуки](notebooks/) · [Код и эксперименты](code/README.md) · [Результаты](results/)

## Запуск

Из корня репозитория:

```bash
jupyter lab notebooks
make -f code/build/research.mk
```

Для сборки PDF нужны XeLaTeX и latexmk.
