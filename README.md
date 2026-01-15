# AI Telegram Bot

## Деплой на BotHost

### 1. Подготовка файлов
- `bot.py` - основной код бота
- `requirements.txt` - зависимости
- `Dockerfile` - конфигурация Docker

### 2. Настройка переменных окружения
Добавьте в настройках BotHost:
- `BOT_TOKEN` - токен бота от @BotFather
- `PERPLEXITY_KEY` - API ключ Perplexity
- `GENAPI_KEY` - API ключ GenAPI
- `ADMIN_ID` - ваш Telegram ID
- `DATABASE_URL` - строка подключения к PostgreSQL

### 3. База данных
Создайте PostgreSQL базу данных в BotHost или используйте внешнюю.

### 4. Запуск
Загрузите файлы и запустите бота через панель BotHost.

## Функции
- Мультимодельный AI чат
- Генерация изображений через SeedEdit
- Система Premium подписок
- Админ панель
- Загрузка фото через Telegraph.ph