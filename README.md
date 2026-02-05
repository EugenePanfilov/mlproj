[![CI](https://github.com/EugenePanfilov/mlproj/actions/workflows/ci.yml/badge.svg)](https://github.com/EugenePanfilov/mlproj/actions/workflows/ci.yml)


# mlproj — минимальный продовый контур: train → artifacts → batch predict → offline monitoring → CI

Проект демонстрирует “живой” контур бинарной классификации на датасете **Adult** (через `fetch_openml`):  
**обучение** → **сохранение артефактов** → **батч-инференс CSV→CSV** → **оффлайн-мониторинг дрейфа/калибровки** → **тесты/CI**.

Внутри поддерживаются 2 модельных варианта (по конфигу):
- `logreg` (логистическая регрессия)
- `hgb` (HistGradientBoostingClassifier)

Есть калибровка вероятностей и принятие решения с порогом + “зоной ручной проверки” (reject-zone).

---

## Быстрый старт

### 1) Установка зависимостей
Рекомендуется Python 3.11 (работает и на 3.10, если зависимости встали).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Обучение
```bash
python train.py --config configs/config.yaml --run-name exp1
```

Результат:
- создаётся папка `artifacts/<RUN_ID>/`
- сохраняются `model.joblib`, `metrics.json`, `config.lock.yaml`, и др.
- в консоль логируется итоговая сводка метрик и выбранный порог

### 3) Батч-инференс (CSV → CSV)
Подготовьте входной CSV **со всеми обязательными признаками** (см. ниже “Схема входа”).

```bash
python predict.py \
  --model-dir artifacts/LATEST \
  --input-csv input.csv \
  --output-csv out.csv
```

На выходе `out.csv` будет содержать исходные колонки +:
- `proba` — вероятность положительного класса
- `decision` — решение (1/0/-1)
- `decision_reason` — причина (`auto | below_threshold | manual_review`)

### 4) Оффлайн-мониторинг (reference vs current)
Подготовьте два CSV:
- `ref.csv` — reference (эталон)
- `curr.csv` — current (текущие данные)

Опционально можно добавить `target` в каждом CSV, тогда посчитаются метрики калибровки/качества вероятностей на обеих выборках.

```bash
python monitor_run.py \
  --model-dir artifacts/LATEST \
  --reference-csv ref.csv \
  --current-csv curr.csv

echo $?
```

Отчёт будет сохранён в `reports/monitor/<RUN_ID>/`.

---

## Makefile команды

```bash
make install
make train
make predict
make monitor
make test
make clean-artifacts
```

---

## Конфигурация

Основные настройки лежат в `configs/config.yaml`:
- `seed`, `split`, `model`, `calibration`
- `thresholding.decision_threshold` + `thresholding.manual_band` (reject-zone)
- `costs` (для подстройки порога по expected cost)
- `monitoring` (пороги алертов и параметры дрейфа)

При обучении сохраняется **фиксированный снимок** настроек:
- `artifacts/<RUN_ID>/config.lock.yaml`

---

## Схема входа (CSV для predict/monitor)

Ожидаемые признаки (Adult):
- `age`, `workclass`, `fnlwgt`, `education`, `education-num`,
  `marital-status`, `occupation`, `relationship`, `race`, `sex`,
  `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`

Опционально:
- `target` — бинарный таргет (0/1). Нужен для обучения и для “полного” мониторинга калибровки.

Валидация схемы выполняется через `pandera`:
- проверяются обязательные колонки, типы и диапазоны (например `age` 17..99)
- при нарушениях скрипты завершаются с ошибкой (exit code ≠ 0)
- при высокой доле пропусков выводится warning в логах

---

## Структура артефактов

После `train.py` создаётся директория вида:

```
artifacts/
  20260204_1556_nogit_exp1/
    model.joblib
    metrics.json
    config.lock.yaml
    feature_names.json
```

Что внутри:
- `model.joblib` — сериализованный объект модели (препроцессор + модель + калибровка + порог/reject-zone)
- `metrics.json` — итоговые метрики на valid (ROC-AUC, PR-AUC, Brier, LogLoss, ECE, expected_cost, threshold)
- `config.lock.yaml` — полный снимок использованных настроек
- `feature_names.json` — имена финальных признаков после препроцессинга (для мониторинга/аналитики)

### `artifacts/LATEST`
`LATEST` — удобный алиас, который резолвится в последнюю созданную папку артефакта (используется в `predict.py` и `monitor_run.py`).

---

## Оффлайн-мониторинг: что считается и как читать отчёт

Запуск:
```bash
python monitor_run.py --model-dir artifacts/LATEST --reference-csv ref.csv --current-csv curr.csv
```

### Метрики дрейфа (основные)
1) **Data drift**
- `psi_max` — максимальный PSI по top-K признакам (чем больше, тем сильнее дрейф)
- `adv_auc` — adversarial AUC (классификатор отличает current от reference по признакам)

2) **Score drift**
- `score_js` — JS-дивергенция распределения скора (`proba`) между reference и current

3) **Калибровка/качество вероятностей** (если есть `target`)
- `ECE (ref)`, `ECE (cur)` — калибровочная ошибка
- `Brier (ref)`, `Brier (cur)` — Brier score
- `ΔECE (cur - ref)`, `ΔBrier (cur - ref)` — разница между current и reference (корректно считается только если `target` есть в обоих наборах)

### Отчёт (reports/monitor/<RUN_ID>/)
В директории отчёта обычно есть:
- `report.md` — Markdown-отчёт со сводкой и таблицей top-фич
- `score_hist.png` — гистограмма распределения `proba` (ref vs cur)
- `psi_top.png` — barplot PSI топ-фич
- `reliability_current.png` — калибровка (если есть `target` в current)

### Severity и коды выхода
Серьёзность определяется правилами из `configs/config.yaml` → секция `monitoring`:

- `OK` → exit code `0`
- `MINOR` → exit code `1`
- `MAJOR` → exit code `2`

Это сделано специально для CI/CD: при `MAJOR` пайплайн может “падать” и требовать внимания.

---

## Как интерпретировать поля решения в predict

`predict.py` добавляет:
- `proba` — вероятность класса 1
- `decision`:
  - `1` — положительный класс (выше порога)
  - `0` — отрицательный класс (ниже порога)
  - `-1` — отправлено на ручную проверку (внутри manual band)
- `decision_reason`:
  - `auto` — решение принято автоматически
  - `below_threshold` — ниже порога, автоматом 0
  - `manual_review` — попали в “зону сомнения” (reject-zone)

---

## Тесты

Запуск:
```bash
pytest -q
```

Минимальный набор проверяет:
- smoke: train → serialize → predict без ошибок
- устойчивость к unseen категориям/NaN
- детерминизм предсказаний после загрузки артефакта

---

## Примечания

- Для загрузки Adult через `fetch_openml` нужен интернет.
- Ненулевой код выхода мониторинга (1/2) — это **не “краш”**, а сигнал MINOR/MAJOR для автоматизации.
