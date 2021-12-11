import logging

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

import gensim
import joblib

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from email_preproccessing import lemmatize_text, del_punct_symbols, del_stop_words

np.random.seed(432)
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

model_d2v = gensim.models.doc2vec.Doc2Vec.load('model_weights/doc2vec_model')
classifier = joblib.load('model_weights/lof_model.sav')

def get_prediction(email_text: str):
    test_bodies = lemmatize_text(del_stop_words(del_punct_symbols([email_text]), stop_words), lemmatizer)
    X_test = np.array([model_d2v.infer_vector(vec) for vec in test_bodies])
    res = classifier.predict(X_test)
    res[res==-1] = 0
    return res[0]

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\! Этот бот проверяет насколько подозрительным является email.',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Отправь текст письмо в чат и получи оценку того, насколько это сообщение легимно')


def form_reply(update: Update, context: CallbackContext) -> None:
    pred = get_prediction(update.message.text)
    response_text = "Текст не вызывает подозрения" if pred == 1 else "Этот текст подозрителен!"
    update.message.reply_text(response_text)


def main() -> None:
    updater = Updater("!!!!")
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, form_reply))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
