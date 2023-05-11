import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import joblib
from haystack.document_stores import InMemoryDocumentStore
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 0")


document_store = InMemoryDocumentStore(use_bm25=True)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1")

doc_dir = "../data/Bible_trained5"
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2")
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 3")

files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 4")

from haystack.nodes import BM25Retriever
from haystack.nodes import DensePassageRetriever
retriever = BM25Retriever(document_store=document_store)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 5")

from haystack.nodes import FARMReader
#*** worked nice in English deepset/roberta-base-squad2
#AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru 
#distilbert-base-cased-distilled-squad
#xlm-roberta-base ----amharic
reader = FARMReader(model_name_or_path="AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru", use_gpu=True)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 6")

from haystack.pipelines import ExtractiveQAPipeline

pipe = ExtractiveQAPipeline(reader, retriever)
from haystack.utils import print_answers
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7")

#print_answers(
#    prediction,
#    details="medium" ## Choose from `minimum`, `medium`, and `all`
#)


def createVerse2(verse_after_decimal_point):
    if verse_after_decimal_point == 1:
        return str (verse_after_decimal_point) + "-" + str (verse_after_decimal_point + 1)
    else:
        return str (verse_after_decimal_point - 1) + "-" + str (verse_after_decimal_point + 1)
    

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 8")


def findTriguame(verse_start):
    varse = "empty"
    
    with open('../data/Bible_verse_with_triguame_full.txt', 'r') as file:
        data = file.read()
    triguame_start = 0
    while data[verse_start:verse_start+12] != 'triguame_end':
            #print("TTTTTTTTT ",data[verse_start:verse_start+14])
            if data[verse_start:verse_start+14] == "triguame_start":
                triguame_start = verse_start
            
            verse_start = verse_start + 1
    
    return data[triguame_start+15:verse_start]
    
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 9")
    

def findVerse2(searchInput):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 10")

    print("<<<<<<<<<<<<<<<<<<<",searchInput)
    varse = "empty"
    triguame_result = ""
    with open('../data/Bible_verse_with_triguame_full.txt', 'r') as file:
        data = file.read()
    
    
    searchResult = data.find(searchInput)
    tabIndex = 0
    endOfVerse = 0
    decimal_point = 0
    if searchResult != -1:
        while data[searchResult:searchResult+11] != 'verse_start':
            if data[searchResult:searchResult+12] == 'verse_number':
                tabIndex = searchResult
                endOfVerse = tabIndex
            if data[searchResult:searchResult+11] == 'verse_point':
                decimal_point = searchResult
            searchResult = searchResult - 1
        #verse string and number    
        varse_number = data[searchResult+12:decimal_point]

        verse_after_decimal_point = int(data[decimal_point+11:tabIndex-1])
        #first verse
        searchResult = searchResult - 1
        endOfFirstVerse = searchResult
        
        first_verse_end = 0
        while data[searchResult:searchResult+12] != 'verse_number':
            #print("TTTTTTTTT ",data[searchResult:searchResult+9])
            if data[searchResult:searchResult+9] == "verse_end":
                first_verse_end = searchResult
            searchResult = searchResult - 1
        varse_sentence_1 = data[searchResult+15: first_verse_end-1]    
        
        #second Verse  
        while data[endOfVerse:endOfVerse+9] != 'verse_end':
            endOfVerse = endOfVerse + 1    
            
        varse_sentence_2 = data[tabIndex+15:endOfVerse-11]
        verse_start = tabIndex
        endOfVerse = endOfVerse+1
        second_tab = 0
        #third verse
        while data[endOfVerse:endOfVerse+9] != 'verse_end':
            if data[endOfVerse:endOfVerse+12] == 'verse_number':
                second_tab = endOfVerse
            endOfVerse = endOfVerse + 1
            
        varse_sentence_3 = data[second_tab+14:endOfVerse-1]
        varse = varse_number + "÷" + createVerse2(verse_after_decimal_point) + ": "+ varse_sentence_1+" \n"+varse_sentence_2 + "\n" + varse_sentence_3
        triguame_result = findTriguame(verse_start)
        triguame_result = triguame_result.replace("\n","")
    return varse,triguame_result
app = Flask(__name__)
api = Api(app)
class MakePrediction(Resource):
    try:
        
        @staticmethod
        def post():
            posted_data = request.get_json()
            user_question = posted_data['question']
            retriver_top_k = 10
            reader_top_k = 5
            prediction = pipe.run(
            query=user_question,
            params={
                "Retriever": {"top_k": retriver_top_k},
                "Reader": {"top_k": reader_top_k}
                })
            
            #=============================
            print_answers(
                prediction,
                details="medium" ## Choose from `minimum`, `medium`, and `all`
            )

            #=============================


            text = ""
            answerList = []
            verseList = []
            triguameList = []
            for x in range(reader_top_k):
                new_context = prediction['answers'][x].context 
                context_list = new_context.split('\n')
                context_list.sort(key=len, reverse=True)
                print(f"Answer: '{prediction['answers'][x].answer}', score: {round(prediction['answers'][x].score, 4)}")

                #add to answersList that will returned by API
                answerList.append(prediction['answers'][x].answer)
                outputVarse = ""
                outputTriguame = ""
                for sentence in context_list:
                    sentence = sentence[3:]
                    #print(sentence,"<====")
                    if len(sentence) < 3: continue
                    #outputVarse = findVerse2(sentence)
                    #print(sentence,"<========")
                    sentence = sentence.replace("\n","")
                    #sentence = sentence.replace("»","")
                    #sentence = sentence.replace("«","")
                    outputVarse,outputTriguame = findVerse2(sentence)

                    if outputVarse != "":
                        break
                        
                outputVarse = outputVarse.replace('\t', '')
                print("\t",outputVarse)
                text = text + new_context
                print()
                print("\t«ከትርጓሜ መጻሕፍት»")
                print()
                print(outputTriguame[:400])

                #add triguame to triguame list that will be returned to api
                verseList.append(outputVarse)
                triguameList.append(outputTriguame[:400])
                print()
                print("===============================================================")

            return jsonify({
                'answers': answerList,
                'verses': verseList,
                'triguames':triguameList
            })
    except:
        print("api error main")
try:
    api.add_resource(MakePrediction, '/predict')
except:
    print("api erreor")

if __name__ == '__main__':
    app.run()