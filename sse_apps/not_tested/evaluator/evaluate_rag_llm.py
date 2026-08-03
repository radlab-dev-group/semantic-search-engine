import os
import django
import logging
import argparse
import json
import tqdm
from typing import List, Dict

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
django.setup()

from system.models import OrganisationUser
from chat.controllers.chat import ChatController
from engine.controllers.database.relational_db import RelationalDBController
from engine.controllers.models_logic.generative import GenerativeModelController

def prepare_parser():
    parser = argparse.ArgumentParser(description="Evaluate RAG quality using LLM as a judge.")
    parser.add_argument("-u", "--username", required=True, type=str, help="Username to use for requests")
    parser.add_argument("-c", "--collection", required=True, type=str, help="Collection name")
    parser.add_argument("-i", "--input-json", required=True, type=str, help="Input JSON file with questions and ground truth")
    parser.add_argument("-o", "--output-json", required=True, type=str, help="Output JSON file with evaluation results")
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite", help="Model to use for evaluation")
    return parser

class RAGEvaluator:
    def __init__(self, user: OrganisationUser, collection_name: str, eval_model: str):
        self.user = user
        self.collection = RelationalDBController.get_collection(collection_name, user)
        if not self.collection:
            raise Exception(f"Collection {collection_name} not found")
        
        self.chat_controller = ChatController()
        self.gen_controller = GenerativeModelController()
        self.eval_model = eval_model

    def evaluate_sample(self, question: str, ground_truth: str = None) -> Dict:
        # 1. Get RAG Response
        chat = self.chat_controller.new_chat(
            organisation_user=self.user,
            collection=self.collection,
            options={"use_rag_supervisor": True, "use_intelligent_query_rewrite": True, "rerank_results": True}
        )
        
        # We simulate adding a message to get the response
        history, user_msg = self.chat_controller.add_user_message(
            chat=chat,
            message=question,
            options={"use_rag_supervisor": True, "user_msg_is_rag_question": True, "rerank_results": True},
            search_options={"max_results": 5, "rerank_results": True}
        )
        
        # Get assistant response
        try:
            response_text, _, state, _ = self.chat_controller.generate_assistant_message_cs_rag(
                chat=chat,
                collection=self.collection,
                last_user_message=user_msg,
                history=history,
                options={"use_rag_supervisor": True, "user_msg_is_rag_question": True, "rerank_results": True, "generative_model": "google/gemini-2.5-flash-lite"},
                organisation_user=self.user,
                sse_engin_config_path="configs/milvus_config.json"
            )
        except Exception as e:
            return {"error": str(e)}

        # 2. Extract context used
        context = ""
        if state and state.rag_message_state:
            res = state.rag_message_state.sse_response
            if res and res.detailed_results_json:
                for hit in res.detailed_results_json:
                    context += f"Document: {hit.get('document_name')}\nContent: {hit.get('text_str')}\n\n"

        # 3. Evaluate using LLM-as-a-judge
        eval_results = self._llm_judge(question, response_text, context, ground_truth)
        
        return {
            "question": question,
            "answer": response_text,
            "context": context,
            "ground_truth": ground_truth,
            "evaluation": eval_results
        }

    def _llm_judge(self, question: str, answer: str, context: str, ground_truth: str = None) -> Dict:
        prompt = f"""Oceń jakość odpowiedzi systemu RAG na podstawie podanego kontekstu i pytania.
Pytanie: {question}
Kontekst: {context}
Odpowiedź systemu: {answer}
{f'Oczekiwana odpowiedź (Ground Truth): {ground_truth}' if ground_truth else ''}

Oceń odpowiedź w skali 1-5 w poniższych kategoriach (zwróć tylko JSON):
1. Faithfulness (Wierność): Czy odpowiedź opiera się wyłącznie na podanym kontekście i nie zawiera halucynacji?
2. Answer Relevance (Trafność): Czy odpowiedź bezpośrednio odpowiada na zadane pytanie?
3. Context Precision (Precyzja kontekstu): Czy dostarczony kontekst był istotny dla odpowiedzi?

Format wyjściowy (JSON):
{{"faithfulness": score, "relevance": score, "precision": score, "explanation": "krótkie uzasadnienie"}}
"""
        try:
            eval_str, _ = self.gen_controller.gen_model_controller.conversation_with_local_model(
                history=[],
                last_user_message=prompt,
                options={"max_new_tokens": 300, "temperature": 0.0},
                model_name_path=self.eval_model
            )
            # Find JSON in response
            start_idx = eval_str.find('{')
            end_idx = eval_str.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                return json.loads(eval_str[start_idx:end_idx])
            return {"raw_eval": eval_str}
        except Exception as e:
            return {"error": str(e)}

def main():
    args = prepare_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    
    from system.controllers import SystemController
    user = SystemController.get_organisation_user(username=args.username)
    if not user:
        logging.error(f"User {args.username} not found")
        return

    evaluator = RAGEvaluator(user, args.collection, args.model)
    
    with open(args.input_json, "r") as f:
        test_data = json.load(f)
    
    results = []
    for sample in tqdm.tqdm(test_data, desc="Evaluating"):
        question = sample.get("question")
        gt = sample.get("ground_truth")
        res = evaluator.evaluate_sample(question, gt)
        results.append(res)
        
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Evaluation finished. Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
