import asyncio
import logging
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
import aiofiles
import os
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from config import TOKEN, ELASTIC_HOST
from elastic import ElasticModule, embeddings, llm
from models import UserState


logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN)
dp = Dispatcher()
router = Router()

elastic = ElasticModule(host=ELASTIC_HOST)


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    """
    Handles the /start command.
    Clears user state and starts the name collection process.
    """

    await state.clear()
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Поиск по тексту",
                    callback_data="request"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Загрузить файл",
                    callback_data="file"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Удалить мои файлы",
                    callback_data="delete_data"
                )
            ],
        ]
    )
    await message.answer(
        'Добро пожаловать в бота для поиска информации по вашим текстам! Пожалуйста, выберите действие:',
        reply_markup=markup
    )
    logging.info("New user: %s, try to create index...", message.from_user.username)
    elastic.check_user_db(str(message.from_user.id))


@dp.callback_query(F.data == "menu")
async def cmd_menu(callback: CallbackQuery, state: FSMContext):
    """
    """
    await state.clear()
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Поиск по тексту",
                    callback_data="request"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Загрузить файл",
                    callback_data="file"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Удалить мои файлы",
                    callback_data="delete_data"
                )
            ],
        ]
    )
    elastic.check_user_db(str(callback.from_user.id))
    await callback.message.answer('Выберите действие:', reply_markup=markup)
    await callback.answer()


@dp.callback_query(F.data == "delete_data")
async def delete_data(callback: CallbackQuery):
    """
    """
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Меню", callback_data="menu"
                )
            ],
        ]
    )
    await callback.message.answer(
        "Ваши файлы были удалены!",
        reply_markup=markup
    )
    elastic.clear_index(str(callback.from_user.id))
    await callback.answer()


@dp.callback_query(F.data == "file")
async def file_load(callback: CallbackQuery, state: FSMContext):
    """
    """
    await state.clear()
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Меню", callback_data="menu"
                )
            ],
        ]
    )
    await callback.message.answer('Прикрепите файл! Максимальный размер 5Мб. Формат TXT', reply_markup=markup)
    await callback.answer()
    await state.set_state(UserState.waiting_for_file)


@router.message(UserState.waiting_for_file)
async def handle_user_file(message: Message, state: FSMContext):
    document = message.document
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Меню", callback_data="menu"
                )
            ],
        ]
    )
    if not document:
        await message.answer("Пожалуйста, отправьте текстовый файл в формате .txt", reply_markup=markup)
        return

    if document.mime_type != "text/plain":
        await message.answer("Неверный формат файла. Поддерживается только .txt", reply_markup=markup)
        return

    if document.file_size > 5 * 1024 * 1024:
        await message.answer("Размер файла превышает 5 Мб.", reply_markup=markup)
        return

    tg_file = await message.bot.get_file(document.file_id)
    file_path = f"tmp/{message.from_user.id}_____{document.file_name}"
    file_data = await message.bot.download_file(tg_file.file_path)
    os.makedirs("tmp", exist_ok=True)
    with open(file_path, "w") as file:
        pass
    async with aiofiles.open(file_path, 'wb') as out_file:
        await out_file.write(file_data.read())

    try:
        await elastic.add_text_file(message.from_user.id, file_path, chunk_size=100)
        await message.answer("✅ Файл успешно загружен и разбит на документы.", reply_markup=markup)
    except Exception as e:
        await message.answer(f"❌ Ошибка при обработке файла: {e}", reply_markup=markup)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@dp.callback_query(F.data == "request")
async def handle_request_callback(callback: CallbackQuery, state: FSMContext):
    """
    """
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Меню", callback_data="menu"
                )
            ],
        ]
    )

    try:
        await state.set_state(UserState.waiting_for_request)
        await callback.message.answer(f"Введите Ваш запрос!", reply_markup=markup)
        await callback.answer()
    except Exception as e:
        await callback.message.answer("Непредвиденная ошибка!")
        raise e


@router.message(UserState.waiting_for_request)
async def handle_user_input(message: Message):
    text = message.text
    user_id = str(message.from_user.id)
    markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Меню", callback_data="menu"
                )
            ],
        ]
    )
    try:
        embedding = embeddings.embed_query(text)

        docs_vector = elastic.search_documents_vector(user_id, embedding, top_k=7)
        docs_text = elastic.search_documents_text(user_id, text, top_k=2)
        combined = list(dict.fromkeys(docs_vector + docs_text))
        logging.info(f"Формируем запрос к LLM с {len(combined)} документами и вопросом: {text[:100]}...")

        docs_text = "\n\n".join(combined)
        prompt = (
            "Ты — помощник, который отвечает на вопросы на основе предоставленных документов.\n"
            "Используй только факты из документов. Если ответа нет — скажи, что не знаешь.\n"
            "Отвечай только на русском языке.\n\n"
            f"Документы:\n{docs_text}\n\n"
            f"Вопрос:\n{text.strip()}"
        )

        result = llm.invoke(prompt)
        logging.info(result)
        answer = str(result)
        response_text = f'Ответ модели:\n{answer}\n\n=========================\n\n'

        for chunk in combined[:3]:
            response_text += f'{chunk}\n\n=========================\n\n'

        await message.answer(response_text, reply_markup=markup)
    except Exception as e:
        await message.answer("Непредвиденная ошибка!")
        raise e


async def main():
    """
    Main entry point to start the bot.
    Includes the router and starts polling.
    """
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
