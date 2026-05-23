### Как запустить дашборд и открыть его публично

#### 1️⃣ Установка зависимостей (один раз)
```powershell
cd C:\\Coding\\Python\\NM\\project
# Если виртуальное окружение ещё не активировано
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

#### 2️⃣ Запуск локального дашборда
```powershell
streamlit run dashboard.py
```
Откроется браузер по адресу `http://localhost:8501`. Все параметры находятся в боковой панели.

#### 3️⃣ Публичный доступ через **ngrok**
1. Установите ngrok (один раз) и добавьте токен:
```powershell
ngrok config add-authtoken <YOUR_TOKEN>
```
2. Запустите скрипт, который поднимает Streamlit и открывает туннель:
```powershell
python share_ngrok.py
```
Скрипт запустит `dashboard.py` в фоне, откроет туннель `ngrok http 8501` и выведет публичный URL, например `https://abcd1234.ngrok.io`. Прервите процесс `Ctrl+C` — оба процесса завершатся.

#### 4️⃣ Публичный доступ через **localtunnel** (альтернатива)
```powershell
python share_dashboard.py
```
Скрипт запустит Streamlit и откроет `localtunnel` (npx). В консоли появится публичный URL вида `https://xxxx.loca.lt`. Прервите процесс `Ctrl+C`.

#### 5️⃣ Что делают файлы проекта
- **dashboard.py** – основное Streamlit‑приложение.
- **share_ngrok.py** – запуск дашборда и публикация через ngrok.
- **share_dashboard.py** – запуск дашборда и публикация через localtunnel.
- **patch_dashboard.py** – одноразовый скрипт‑патч, уже не нужен (можно удалить).

#### 6️⃣ Остановка
Нажмите `Ctrl+C` в терминале, где запущен любой из скриптов (Streamlit, ngrok, localtunnel). Скрипт выведет сообщение о завершении.

---
*Все команды следует выполнять из каталога `C:\\Coding\\Python\\NM\\project`.*