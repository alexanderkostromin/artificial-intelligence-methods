import telebot
from tensorflow import keras
from keras.utils import load_img, img_to_array
from PIL import Image as img
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TOKEN = "6188269355:AAGhDSWM2k2mpUYZyOZgGhXFQZpYB4yAN2A"

bot = telebot.TeleBot(TOKEN)

model_cancer_detektor = keras.models.load_model('skin_cancer.h5')

types = {
    0: 'Меланоцитарный невус',
    1: 'Меланома',
    2: 'доброкачественный кератоз',
    3: 'Базально-клеточный рак',
    4: 'Актинический кератоз и интраэпителиальная карцинома',
    5: 'Сосудистые поражения',
    6: 'Дерматофиброма',
}


@bot.message_handler(commands=['start'])
def start_handler(message):
    bot.send_message(message.chat.id, "Приветствую вас в боте для распознавания рака кожи!".format(message.from_user, bot.get_me()),
                     parse_mode='html')


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        print(file_info)
        downloaded_file = bot.download_file(file_info.file_path)
        src = file_info.file_path

        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.reply_to(message, 'Фото загружено')
        photo = load_img(src, target_size=(80, 80))
        x = img_to_array(photo)
        x /= 255
        print(4)
        p = x.reshape([1, 80, 80, 3])
        print(5)
        results = [round(x, 3) for x in model_cancer_detektor.predict(p)[0]]
        max_res = results[0]
        ind_max = 0

        for i in range(1, len(results)):
            if results[i] > max_res:
                max_res = results[i]
                ind_max = i

        back_message = ''
        if max_res >= 0.5:
            back_message = f'С вероятностью {max_res:0.3} у вас {types[ind_max]}'
        else:
            back_message = 'Нейронная сеть не смогла точно определить тип вашего кожного заболевания. Пожалуйста обратитесь к врачу!'
        bot.send_message(message.chat.id,
                         back_message.format(
                             message.from_user, bot.get_me()), parse_mode='html')
        os.remove(src)
    except Exception as e:
        bot.reply_to(message, e)


if __name__ == '__main__':
    bot.polling(none_stop=True)