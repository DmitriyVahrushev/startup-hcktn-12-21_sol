import gensim
import joblib

model_d2v = gensim.models.doc2vec.Doc2Vec.load('model_weights/doc2vec_model')
classifier = joblib.load('model_weights/lof_model.sav')