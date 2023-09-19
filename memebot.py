import asyncio
import json
import logging
import os
import numpy as np

from aiogram import Bot, types, Dispatcher, F
import aiohttp  # noqa: F401
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters.callback_data import CallbackData

from preprocess import MEM_EMBEDDINGS, MEM_CAPTURES, NUM_MEMS, NUM_TO_SHOW, CURRENT_USER_DATA, MEM_LIKES,\
    USER_EMBEDDINGS, USER_ID2NUM, MEM_IDS, EMB_SIZE, MEM_DISLIKES

dp = Dispatcher()
bot = Bot(token=os.environ['BOT_TOKEN'])


@dp.message(Command('start', prefix='/'))
@dp.message(Command('help', prefix='/'))
async def send_welcome(message: types.Message):
    await message.answer(
        "Привет!\nЯ Мемный Бот!\n"
        "Я помогу тебе найти любой мем! Ты можешь отправить мне:\n"
        "1. Текстовый запрос\n"
        "2. /for_you - мемы заботливо отобранные специально для тебя\n"
        "3. /popular - самые популярные мемы из нашей базы\n"
        "4. /feedback - мы будем тебе очень благодарны если ты заполнишь короткую форму\n"
        "5. Картинку мема для добавления в нашу базу"
    )


@dp.message(Command('feedback', prefix='/'))
async def send_welcome(message: types.Message):
    await message.answer("Пока здесь ничего нет, но скоро появится форма для вашего фидбека!")


@dp.message(Command('save', prefix='/'))
async def save(message: types.Message):
    np.save('statsfiles/user_embs.npy', USER_EMBEDDINGS)
    with open('statsfiles/usrid2num.json', 'w') as f:
        f.write(json.dumps(USER_ID2NUM))
    with open('statsfiles/memids.txt', 'w') as f:
        f.write(str(MEM_IDS))
    with open('statsfiles/memlikes.txt', 'w') as f:
        f.write(str(MEM_LIKES))
    with open('statsfiles/memdislikes.txt', 'w') as f:
        f.write(str(MEM_DISLIKES))
    await message.answer('успешно сохранено')


class MemCallbackFactory(CallbackData, prefix="fabmem"):
    action: str
    user_id: int
    mem_num: int


def get_keyboard_for_search(user_id, mem_num):
    builder = InlineKeyboardBuilder()
    builder.button(text="👍", callback_data=MemCallbackFactory(action='search_like', user_id=user_id, mem_num=mem_num))
    builder.button(text="👎", callback_data=MemCallbackFactory(action='search_dislike', user_id=user_id, mem_num=mem_num))
    builder.button(text='next', callback_data=MemCallbackFactory(action='search_next', user_id=user_id, mem_num=mem_num))
    builder.adjust(2)
    return builder.as_markup()


def get_keyboard_for_look(user_id, mem_num):
    builder = InlineKeyboardBuilder()
    builder.button(text="👍", callback_data=MemCallbackFactory(action='look_like', user_id=user_id, mem_num=mem_num))
    builder.button(text="👎", callback_data=MemCallbackFactory(action='look_dislike', user_id=user_id, mem_num=mem_num))
    builder.button(text='next', callback_data=MemCallbackFactory(action='look_next', user_id=user_id, mem_num=mem_num))
    builder.adjust(2)
    return builder.as_markup()


def get_data_for_user(mode, user_id, query=None):
    if str(user_id) not in USER_ID2NUM.keys():
        USER_ID2NUM[str(user_id)] = len(USER_EMBEDDINGS)
        USER_EMBEDDINGS.append(np.zeros(EMB_SIZE))

    if mode == 'popular':
        choice = np.random.choice(NUM_MEMS, NUM_TO_SHOW, replace=False)
        nums = sorted(choice, key=lambda i: MEM_LIKES[i])
        return nums
    if mode == 'for_you':
        user_emb = USER_EMBEDDINGS[USER_ID2NUM[str(user_id)]]
        choice = np.random.choice(NUM_MEMS, NUM_TO_SHOW, replace=False)
        mem_embs = MEM_EMBEDDINGS[choice]
        idxs = np.argsort(mem_embs @ user_emb.T)
        return choice[idxs]
    else:
        revs = [len(query.intersection(MEM_CAPTURES[i])) for i in range(NUM_MEMS)]
        nums = np.argsort(revs)
        return nums[-NUM_TO_SHOW:]


@dp.message(Command('popular', prefix='/'))
@dp.message(Command('for_you', prefix='/'))
async def look_mem(message: types.Message, user_id=None):
    if user_id is None:
        user_id = message.from_user.id
        CURRENT_USER_DATA[user_id] = get_data_for_user(mode=message.text[1:], user_id=user_id)

    user_data = CURRENT_USER_DATA[user_id]
    cur_display = user_data[-1]
    CURRENT_USER_DATA[user_id] = user_data[:-1]

    if MEM_IDS[cur_display] != -1:
        mem = MEM_IDS[cur_display]
    elif f'{cur_display}.jpg' in os.listdir('images'):
        mem = types.FSInputFile(f"images/{cur_display}.jpg")
    else:
        mem = types.FSInputFile(f"images/{cur_display}.png")
    res = await message.answer_photo(mem, reply_markup=get_keyboard_for_look(user_id, cur_display))
    MEM_IDS[cur_display] = res.photo[-1].file_id


@dp.message(F.text)
async def find_mem(message: types.Message, user_id=None):
    if user_id is None:
        user_id = message.from_user.id
        logging.info(f'user {user_id} query {message.text.lower()}')
        CURRENT_USER_DATA[user_id] = get_data_for_user(
            mode='search', user_id=user_id, query=set(message.text.lower().strip().split())
        )

    user_data = CURRENT_USER_DATA[user_id]
    cur_display = user_data[-1]
    CURRENT_USER_DATA[user_id] = user_data[:-1]

    if MEM_IDS[cur_display] != -1:
        mem = MEM_IDS[cur_display]
    elif f'{cur_display}.jpg' in os.listdir('images'):
        mem = types.FSInputFile(f"images/{cur_display}.jpg")
    else:
        mem = types.FSInputFile(f"images/{cur_display}.png")
    res = await message.answer_photo(mem, reply_markup=get_keyboard_for_search(user_id, cur_display))
    MEM_IDS[cur_display] = res.photo[-1].file_id


@dp.message(F.photo)
async def add_mem(message: types.Message):
    mem_num = NUM_MEMS + len(os.listdir('pending'))
    await bot.download(message.photo[-1], f'./pending/{mem_num}.jpg')
    await message.answer('Мы добавили твой мем в очередь и скоро внесем его в нашу базу')


@dp.message()
async def end_search(message: types.Message):
    i = np.random.randint(0, 3)
    if i == 0:
        await message.answer("Упс, кажется это все, что мы смогли найти")
    if i == 1:
        await message.answer("Кажется кому-то пора дальше работать")
    if i == 2:
        await message.answer("Все кончились хиханьки хаханьки")


@dp.callback_query(MemCallbackFactory.filter())
async def callbacks_num(callback: types.CallbackQuery, callback_data: MemCallbackFactory):
    mode, action = callback_data.action.split('_')
    user_id = callback_data.user_id
    mem_num = callback_data.mem_num

    if action == 'like':
        MEM_LIKES[mem_num] += 1
        cur_usr_emb = USER_EMBEDDINGS[USER_ID2NUM[str(user_id)]]
        cur_mem_emb = MEM_EMBEDDINGS[mem_num]
        cur_usr_emb += cur_mem_emb
        USER_EMBEDDINGS[USER_ID2NUM[str(user_id)]] = cur_usr_emb / np.linalg.norm(cur_usr_emb)
        logging.info(f'user {user_id} liked {mem_num}')
    elif action == 'dislike':
        MEM_DISLIKES[mem_num] += 1
        logging.info(f'user {user_id} disliked {mem_num}')
    else:
        logging.info(f'user {user_id} next {mem_num}')

    await callback.message.delete_reply_markup()
    if not len(CURRENT_USER_DATA[user_id]):
        await end_search(callback.message)
    elif mode == 'search':
        await find_mem(callback.message, user_id)
    elif mode == 'look':
        await look_mem(callback.message, user_id)

    await callback.answer()


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='bot.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    asyncio.run(main())
