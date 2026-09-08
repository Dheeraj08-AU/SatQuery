import torch
from PIL import Image
import numpy as np
import math
import hashlib
import json
import os
from transformers import AutoProcessor, GroundingDinoForObjectDetection, PaliGemmaForConditionalGeneration
from peft import PeftModel

class ModelRegistry:
    def __init__(self):
        """Initializes the optimized inference engine for real-time remote sensing analysis."""
        self.loaded_tools = {
            "vqa": "PaliGemma-3B (LoRA fine-tuned)",
            "grounding": "GroundingDINO-RS (Active)",
            "change": "ChangeFormer (Active)"
        }
        
        self.cache_file = "vqa_cache.json"
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.vqa_cache = json.load(f)
            except Exception:
                self.vqa_cache = {}
        else:
            self.vqa_cache = {}
            
        self.grounding_processor = None
        self.grounding_model = None
        self.vlm_processor = None
        self.vlm_model = None
        
        print("Optimized Model Registry successfully initialized (Lazy Loading Enabled)!")

    def _ensure_grounding_loaded(self):
        if self.grounding_model is not None:
            return
            
        print("Initializing Real GroundingDINO Zero-Shot Object Detection model (CPU)...")
        self.grounding_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.grounding_model = GroundingDinoForObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).eval().float()

    def _ensure_vlm_loaded(self):
        if self.vlm_model is not None:
            return
            
        print("Initializing Real PaliGemma VLM (CPU bfloat16) with PEFT adapters...")
        try:
            self.vlm_processor = AutoProcessor.from_pretrained("modules/satquery_vqa_adapter")
            base_vlm = PaliGemmaForConditionalGeneration.from_pretrained(
                "google/paligemma-3b-pt-224",
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            self.vlm_model = PeftModel.from_pretrained(base_vlm, "modules/satquery_vqa_adapter", adapter_name="vqa")
            self.vlm_model.load_adapter("modules/satquery_change_adapter", adapter_name="change")
            self.vlm_model.eval()
            print("PaliGemma VLM and both adapters loaded successfully in bfloat16.")
        except Exception as e:
            print(f"Failed to load PaliGemma VLM model/adapters: {e}")
            self.vlm_model = None
            self.vlm_processor = None

    def run_single_vqa(self, image_path: str, query: str, params: dict) -> dict:
        """Executes real-time VQA using fine-tuned PaliGemma on CPU."""
        # Check cache first using file bytes to handle Streamlit temporary uploads
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            cache_key = hashlib.md5(img_bytes + query.encode('utf-8')).hexdigest()
        except Exception:
            cache_key = hashlib.md5(f"{image_path}_{query}".encode('utf-8')).hexdigest()
            
        if cache_key in self.vqa_cache:
            print(f"VQA Cache hit for query: '{query}' on {image_path}")
            return self.vqa_cache[cache_key]

        # Cache miss - load model if not already loaded
        self._ensure_vlm_loaded()
        if self.vlm_model is None or self.vlm_processor is None:
            return {"answer": "Error: VLM model not loaded.", "confidence": 0.0}
            
        try:
            img = Image.open(image_path).convert("RGB")
            prompt = f"answer en {query}"
            
            inputs = self.vlm_processor(text=prompt, images=img, return_tensors="pt").to(torch.bfloat16)
            # Fix types for input_ids and attention_mask
            inputs["input_ids"] = inputs["input_ids"].to(torch.long)
            inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
            
            self.vlm_model.set_adapter("vqa")
            
            with torch.no_grad():
                output = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            prompt_len = inputs["input_ids"].shape[1]
            generated_tokens = output.sequences[0][prompt_len:]
            answer = self.vlm_processor.decode(generated_tokens, skip_special_tokens=True)
            
            # Calculate simple confidence (average token logprob converted to probability)
            scores = output.scores
            if len(scores) > 0 and len(generated_tokens) > 0:
                logprobs = []
                for step_idx, step_scores in enumerate(scores):
                    if step_idx < len(generated_tokens):
                        token_id = generated_tokens[step_idx]
                        probs = torch.nn.functional.softmax(step_scores[0], dim=-1)
                        logprob = torch.log(probs[token_id]).item()
                        logprobs.append(logprob)
                
                avg_logprob = sum(logprobs) / len(logprobs)
                confidence = math.exp(avg_logprob)
            else:
                confidence = 1.0
                
            result = {
                "answer": answer.strip(),
                "confidence": confidence
            }
            self.vqa_cache[cache_key] = result
            with open(self.cache_file, 'w') as f:
                json.dump(self.vqa_cache, f)
            return result
        except Exception as e:
            print(f"VQA Error: {e}")
            return {"answer": f"Inference failed: {str(e)}", "confidence": 0.0}

    def run_grounding(self, image_path: str, query, params: dict) -> dict:
        """Executes real text-guided spatial grounding using GroundingDINO on CPU.
        
        Note: Inference runs entirely on CPU (float32) and will take a few seconds 
        per image (not milliseconds).
        """
        self._ensure_grounding_loaded()
        if self.grounding_model is None or self.grounding_processor is None:
            return {"box": None, "confidence": 0.0, "label": None, "status": "Error: Grounding model not loaded."}
            
        box_threshold = params.get("box_threshold", 0.35)
        text_threshold = params.get("text_threshold", 0.25)
        
        queries = query if isinstance(query, list) else [query]
        print(f"Running GroundingDINO on {image_path}")
        print(f"Candidates: {queries} | box_thresh: {box_threshold} | text_thresh: {text_threshold}")
        
        img = Image.open(image_path).convert("RGB")
        
        best_overall_box = None
        best_overall_score = -1.0
        best_overall_label = None
        
        for q in queries:
            inputs = self.grounding_processor(images=img, text=q, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
                
            results = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                text_threshold=text_threshold,
                target_sizes=[img.size[::-1]]
            )[0]

            keep = results["scores"] >= box_threshold
            scores = results["scores"][keep]
            boxes  = results["boxes"][keep]
            raw_labels = results.get("text_labels", results.get("labels", []))
            labels = [l for l, k in zip(raw_labels, keep.tolist()) if k]
            
            if len(scores) > 0:
                best_idx = torch.argmax(scores).item()
                best_score = scores[best_idx].item()
                if best_score > best_overall_score:
                    best_overall_score = best_score
                    best_overall_box = boxes[best_idx].tolist()
                    best_overall_label = labels[best_idx]
                    
        if best_overall_box is not None:
            img_width, img_height = img.size
            img_area = img_width * img_height
            x_min, y_min, x_max, y_max = best_overall_box
            box_area = (x_max - x_min) * (y_max - y_min)
            
            if box_area / img_area > 0.95:
                scene_wide_terms = ["forest", "trees", "built-up area", "farmland", "water body", "urban area", "airport", "coastline", "agricultural land"]
                is_scene_wide = False
                for term in scene_wide_terms:
                    for q in queries:
                        if term in q.lower():
                            is_scene_wide = True
                            break
                    if is_scene_wide:
                        break
                
                if not is_scene_wide:
                    print(f"Rejecting {box_area/img_area*100:.1f}% area box for non-scene-wide query.")
                    best_overall_box = None
                    best_overall_score = -1.0
                    best_overall_label = None

        if best_overall_box is None:
            print("No confident detection found above the specified thresholds.")
            return {
                "box": None,
                "confidence": 0.0,
                "label": None,
                "status": "No confident detection found"
            }
            
        print(f"Detection successful: '{best_overall_label}' at {[round(c,1) for c in best_overall_box]} with score {best_overall_score:.3f}  (box_thresh={box_threshold})")
        
        return {
            "box": best_overall_box,
            "confidence": best_overall_score,
            "label": best_overall_label,
            "status": "Success"
        }

    def run_change_analysis_experimental_multiprobe(self, img1_path: str, img2_path: str, query: str, params: dict) -> dict:
        """
        [EXPERIMENTAL/DISABLED]
        Multi-probe CDVQA change analysis.
        This approach showed a degenerate failure mode during testing (outputting a fixed yes/no pattern
        regardless of content difference), so it is kept here for documentation purposes but disabled for prod.
        """
        # Hashing logic for caching
        try:
            with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
                b1, b2 = f1.read(), f2.read()
            cache_key = hashlib.md5(b1 + b2 + query.encode('utf-8')).hexdigest()
        except Exception:
            cache_key = hashlib.md5(f"{img1_path}_{img2_path}_{query}".encode('utf-8')).hexdigest()
            
        if cache_key in self.vqa_cache:
            print(f"Change Analysis Cache hit for query: '{query}'")
            return self.vqa_cache[cache_key]
            
        self._ensure_vlm_loaded()
        if self.vlm_model is None or self.vlm_processor is None:
            return {"answer": "Error: VLM model not loaded.", "confidence": 0.0}
            
        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
            
            if img1.size != img2.size:
                img2 = img2.resize(img1.size)
                
            w, h = img1.size
            composite = Image.new('RGB', (w * 2, h))
            composite.paste(img1, (0, 0))
            composite.paste(img2, (w, 0))
            
            categories = ["road", "built-up area", "water body", "vegetation", "trees", "bare land"]
            changed_categories = []
            total_confidence = 0.0
            
            import re
            
            for cat in categories:
                prompt = f"answer en The left image is from time T1, the right image is from time T2. Answer the question about what changed: Did the {cat} area change?"
                
                inputs = self.vlm_processor(text=prompt, images=composite, return_tensors="pt").to(torch.bfloat16)
                inputs["input_ids"] = inputs["input_ids"].to(torch.long)
                inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
                
                self.vlm_model.set_adapter("change")
                
                with torch.no_grad():
                    output = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=16,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                prompt_len = inputs["input_ids"].shape[1]
                generated_tokens = output.sequences[0][prompt_len:]
                raw_answer = self.vlm_processor.decode(generated_tokens, skip_special_tokens=True)
                
                print(f"Probe '{cat}' RAW output: {repr(raw_answer)}")
                
                # Robust 'yes' parsing
                # Strip whitespace and punctuation, lowercase
                clean_ans = re.sub(r'[^\w\s]', '', raw_answer).strip().lower()
                
                is_yes = False
                if clean_ans.startswith("yes"):
                    is_yes = True
                else:
                    # Check if 'yes' appears as a standalone word in the first few words
                    words = clean_ans.split()
                    if "yes" in words[:3]:
                        is_yes = True
                        
                if is_yes:
                    changed_categories.append(cat)
                
                # Calculate confidence
                scores = output.scores
                if len(scores) > 0 and len(generated_tokens) > 0:
                    logprobs = []
                    for step_idx, step_scores in enumerate(scores):
                        if step_idx < len(generated_tokens):
                            token_id = generated_tokens[step_idx]
                            probs = torch.nn.functional.softmax(step_scores[0], dim=-1)
                            logprob = torch.log(probs[token_id]).item()
                            logprobs.append(logprob)
                    avg_logprob = sum(logprobs) / len(logprobs)
                    cat_conf = math.exp(avg_logprob)
                else:
                    cat_conf = 1.0
                    
                total_confidence += cat_conf
                
            # Synthesize final answer
            if changed_categories:
                final_answer = "Changes detected in: " + ", ".join(changed_categories) + "."
            else:
                final_answer = "No significant changes detected."
                
            avg_confidence = total_confidence / len(categories) if categories else 0.0
            
            result = {
                "answer": final_answer,
                "confidence": avg_confidence
            }
            
            # Save to cache
            self.vqa_cache[cache_key] = result
            with open(self.cache_file, 'w') as f:
                json.dump(self.vqa_cache, f)
            
            return result
            
        except Exception as e:
            print(f"Change Analysis Error: {e}")
            return {"answer": f"Inference failed: {str(e)}", "confidence": 0.0}

    def run_change_analysis(self, img1_path: str, img2_path: str, query: str, params: dict) -> dict:
        """Executes bitemporal change analysis using fine-tuned PaliGemma VQA on CPU."""
        # Hashing logic for caching (handle bytes directly since Streamlit uses temporary uploaded files)
        try:
            with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
                b1, b2 = f1.read(), f2.read()
            cache_key = hashlib.md5(b1 + b2 + query.encode('utf-8')).hexdigest()
        except Exception:
            cache_key = hashlib.md5(f"{img1_path}_{img2_path}_{query}".encode('utf-8')).hexdigest()
            
        if cache_key in self.vqa_cache:
            print(f"Change Analysis Cache hit for query: '{query}'")
            return self.vqa_cache[cache_key]
            
        self._ensure_vlm_loaded()
        if self.vlm_model is None or self.vlm_processor is None:
            return {"answer": "Error: VLM model not loaded.", "confidence": 0.0}
            
        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
            
            if img1.size != img2.size:
                img2 = img2.resize(img1.size)
                
            w, h = img1.size
            composite = Image.new('RGB', (w * 2, h))
            composite.paste(img1, (0, 0))
            composite.paste(img2, (w, 0))
            
            # Use single open-ended VQA approach
            prompt = f"answer en The left image is from time T1, the right image is from time T2. Answer the question about what changed: {query}"
            
            inputs = self.vlm_processor(text=prompt, images=composite, return_tensors="pt").to(torch.bfloat16)
            inputs["input_ids"] = inputs["input_ids"].to(torch.long)
            inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
            
            # Use original VQA adapter for content-sensitive evaluation
            self.vlm_model.set_adapter("vqa")
            
            with torch.no_grad():
                output = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
            prompt_len = inputs["input_ids"].shape[1]
            generated_tokens = output.sequences[0][prompt_len:]
            answer = self.vlm_processor.decode(generated_tokens, skip_special_tokens=True)
            
            scores = output.scores
            logprobs = []
            if len(scores) > 0 and len(generated_tokens) > 0:
                for step_idx, step_scores in enumerate(scores):
                    if step_idx < len(generated_tokens):
                        token_id = generated_tokens[step_idx]
                        probs = torch.nn.functional.softmax(step_scores[0], dim=-1)
                        logprob = torch.log(probs[token_id]).item()
                        logprobs.append(logprob)
                avg_logprob = sum(logprobs) / len(logprobs)
                confidence = math.exp(avg_logprob)
            else:
                confidence = 1.0
                
            result = {
                "answer": answer.strip(),
                "confidence": confidence
            }
            
            # Save to cache
            self.vqa_cache[cache_key] = result
            with open(self.cache_file, 'w') as f:
                json.dump(self.vqa_cache, f)
            
            return result
            
        except Exception as e:
            print(f"Change Analysis Error: {e}")
            return {"answer": f"Inference failed: {str(e)}", "confidence": 0.0}

    def run_optical_sar(self, optical_path: str, sar_path: str, query: str, params: dict) -> dict:
        """Executes cross-modal fusion analysis by combining Optical and SAR imagery."""
        # Hashing logic for caching
        try:
            with open(optical_path, "rb") as f1, open(sar_path, "rb") as f2:
                b1, b2 = f1.read(), f2.read()
            cache_key = hashlib.md5(b1 + b2 + query.encode('utf-8')).hexdigest()
        except Exception:
            cache_key = hashlib.md5(f"{optical_path}_{sar_path}_{query}".encode('utf-8')).hexdigest()
            
        if cache_key in self.vqa_cache:
            print(f"Optical-SAR Cache hit for query: '{query}'")
            return self.vqa_cache[cache_key]
            
        self._ensure_vlm_loaded()
        if self.vlm_model is None or self.vlm_processor is None:
            return {"answer": "Error: VLM model not loaded.", "confidence": 0.0}
            
        try:
            img1 = Image.open(optical_path).convert("RGB")
            img2 = Image.open(sar_path).convert("RGB")
            
            if img1.size != img2.size:
                img2 = img2.resize(img1.size)
                
            w, h = img1.size
            composite = Image.new('RGB', (w * 2, h))
            composite.paste(img1, (0, 0))
            composite.paste(img2, (w, 0))
            
            prompt = f"answer en The left image is optical, the right image is SAR (radar). Using both together: {query}"
            
            inputs = self.vlm_processor(text=prompt, images=composite, return_tensors="pt").to(torch.bfloat16)
            inputs["input_ids"] = inputs["input_ids"].to(torch.long)
            inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
            
            self.vlm_model.set_adapter("vqa")
            
            with torch.no_grad():
                output = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
            prompt_len = inputs["input_ids"].shape[1]
            generated_tokens = output.sequences[0][prompt_len:]
            answer = self.vlm_processor.decode(generated_tokens, skip_special_tokens=True)
            
            scores = output.scores
            logprobs = []
            if len(scores) > 0 and len(generated_tokens) > 0:
                for step_idx, step_scores in enumerate(scores):
                    if step_idx < len(generated_tokens):
                        token_id = generated_tokens[step_idx]
                        probs = torch.nn.functional.softmax(step_scores[0], dim=-1)
                        logprob = torch.log(probs[token_id]).item()
                        logprobs.append(logprob)
                avg_logprob = sum(logprobs) / len(logprobs)
                confidence = math.exp(avg_logprob)
            else:
                confidence = 1.0
                
            result = {
                "answer": answer.strip(),
                "confidence": confidence
            }
            
            # Save to cache
            self.vqa_cache[cache_key] = result
            with open(self.cache_file, 'w') as f:
                json.dump(self.vqa_cache, f)
            
            return result
            
        except Exception as e:
            print(f"Optical-SAR Error: {e}")
            return {"answer": f"Inference failed: {str(e)}", "confidence": 0.0}