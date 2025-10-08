"""Telegram bot for text-based search and document management using Elastic and LLM."""

import asyncio
import logging
import os
import aiofiles
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
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


def get_main_menu() -> InlineKeyboardMarkup:
    """Return the main menu inline keyboard."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Поиск по тексту", callback_data="request")],
            [InlineKeyboardButton(text="Загрузить файл", callback_data="file")],
            [InlineKeyboardButton(text="Удалить мои файлы", callback_data="delete_data")],
        ]
    )


def get_back_to_menu() -> InlineKeyboardMarkup:
    """Return a keyboard with a single 'Menu' button."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Меню", callback_data="menu")]]
    )


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext) -> None:
    """
    Handle /start command.
    Clears user state, greets the user, and shows the main menu.
    """
    await state.clear()
    await message.answer(
        (
            "Добро пожаловать в бота для поиска информации по вашим текстам! "
            "Пожалуйста, выберите действие:"
        ),
        reply_markup=get_main_menu(),
    )
    logging.info("New user: %s, trying to create index...", message.from_user.username)
    elastic.check_user_db(str(message.from_user.id))


@dp.callback_query(F.data == "menu")
async def cmd_menu(callback: CallbackQuery, state: FSMContext) -> None:
    """Return user to main menu."""
    await state.clear()
    elastic.check_user_db(str(callback.from_user.id))
    await callback.message.answer("Выберите действие:", reply_markup=get_main_menu())
    await callback.answer()


@dp.callback_query(F.data == "delete_data")
async def delete_data(callback: CallbackQuery) -> None:
    """Handle user request to delete their uploaded files."""
    elastic.clear_index(str(callback.from_user.id))
    await callback.message.answer("Ваши файлы были удалены!", reply_markup=get_back_to_menu())
    await callback.answer()


@dp.callback_query(F.data == "file")
async def file_load(callback: CallbackQuery, state: FSMContext) -> None:
    """Handle file upload request."""
    await state.clear()
    await callback.message.answer(
        "Прикрепите файл! Максимальный размер 5Мб. Формат TXT",
        reply_markup=get_back_to_menu(),
    )
    await callback.answer()
    await state.set_state(UserState.waiting_for_file)


@router.message(UserState.waiting_for_file)
async def handle_user_file(message: Message, state: FSMContext) -> None:
    """Process and store user-uploaded text files."""
    document = message.document
    markup = get_back_to_menu()

    if not document:
        await message.answer("Пожалуйста, отправьте текстовый файл в формате .txt", reply_markup=markup)
        return

    if document.mime_type != "text/plain":
        await message.answer("Неверный формат файла. Поддерживается только .txt", reply_markup=markup)
        return

    if document.file_size > 5 * 1024 * 1024:
        await message.answer("Размер файла превышает 5 Мб.", reply_markup=markup)
        return

    os.makedirs("tmp", exist_ok=True)
    file_path = f"tmp/{message.from_user.id}_____{document.file_name}"

    tg_file = await message.bot.get_file(document.file_id)
    file_data = await message.bot.download_file(tg_file.file_path)

    async with aiofiles.open(file_path, "wb") as out_file:
        await out_file.write(file_data.read())

    try:
        await elastic.add_text_file(message.from_user.id, file_path, chunk_size=100)
        await message.answer("✅ Файл успешно загружен и обработан.", reply_markup=markup)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        await message.answer(f"❌ Ошибка при обработке файла: {exc}", reply_markup=markup)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@dp.callback_query(F.data == "request")
async def handle_request_callback(callback: CallbackQuery, state: FSMContext) -> None:
    """Handle the text search request."""
    await state.set_state(UserState.waiting_for_request)
    await callback.message.answer("Введите ваш запрос!", reply_markup=get_back_to_menu())
    await callback.answer()


@router.message(UserState.waiting_for_request)
async def handle_user_input(message: Message) -> None:
    """Process user text queries and return AI-generated response."""
    text = message.text
    user_id = str(message.from_user.id)

    try:
        embedding = embeddings.embed_query(text)
        docs_vector = elastic.search_documents_vector(user_id, embedding, top_k=7)
        docs_text = elastic.search_documents_text(user_id, text, top_k=2)
        combined = list(dict.fromkeys(docs_vector + docs_text))

        logging.info(
            "Forming LLM request with %d documents and query: %.100s...",
            len(combined),
            text,
        )

        docs_combined = "\n\n".join(combined)
        prompt = (
            "Ты — помощник, который отвечает на вопросы на основе предоставленных документов.\n"
            "Используй только факты из документов. Если ответа нет — скажи, что не знаешь.\n"
            "Отвечай только на русском языке.\n\n"
            f"Документы:\n{docs_combined}\n\nВопрос:\n{text.strip()}"
        )

        result = llm.invoke(prompt)
        response_text = f"Ответ модели:\n{result}\n\n=========================\n\n"

        for chunk in combined[:3]:
            response_text += f"{chunk}\n\n=========================\n\n"

        await message.answer(response_text, reply_markup=get_back_to_menu())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        await message.answer("Непредвиденная ошибка!")
        logging.exception("Error while processing user input: %s", exc)


async def main() -> None:
    """Main entry point to start the bot."""
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
