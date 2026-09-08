import pytest
from unittest.mock import MagicMock
from modules.agent_controller import AgentController, IntentClassification

class TestAgentControllerE2E:
    @pytest.fixture
    def agent(self, monkeypatch):
        # We want to mock Gemini, but let it return messy queries to test parsing
        # Actually, since it's an E2E test, we can mock Gemini's response 
        # or we can just mock `self.client.models.generate_content`
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
        
        # In AgentController.__init__, it will init ModelRegistry
        # We can just patch genai.Client so it doesn't crash on fake key
        from google import genai
        original_client = genai.Client
        
        def fake_client(*args, **kwargs):
            client = MagicMock()
            
            def fake_generate_content(model, contents, config, **kw):
                prompt = contents
                # Determine response based on the messy query in the prompt
                if "messy query 1" in prompt:
                    resp = IntentClassification(
                        task="grounding",
                        target_entity="water body",
                        search_nouns=["water body.", "lake."],
                        classifier_self_reported_confidence=0.9
                    )
                elif "messy query 2" in prompt:
                    resp = IntentClassification(
                        task="grounding",
                        target_entity="built-up area",
                        search_nouns=["building.", "built-up area."],
                        classifier_self_reported_confidence=0.85
                    )
                elif "messy query 3" in prompt:
                    resp = IntentClassification(
                        task="grounding",
                        target_entity="road",
                        search_nouns=["road.", "highway."],
                        classifier_self_reported_confidence=0.95
                    )
                else:
                    resp = IntentClassification(
                        task="vqa",
                        target_entity=None,
                        search_nouns=None,
                        classifier_self_reported_confidence=0.8
                    )
                
                mock_resp = MagicMock()
                mock_resp.text = resp.model_dump_json()
                return mock_resp
                
            client.models.generate_content = fake_generate_content
            return client
            
        monkeypatch.setattr(genai, "Client", fake_client)
        
        return AgentController()

    def test_e2e_messy_query_1(self, agent, tmp_path):
        # Create a fake image so it passes validation and runs grounding
        img_path = tmp_path / "test.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="green").save(img_path)
        
        trace = agent.route_and_configure(
            query="messy query 1: I need you to highlight the water body for me please",
            image_paths=[str(img_path)]
        )
        assert trace.selected_task.value == "Text-Guided Region Grounding"
        # The result from GroundingDINO on a solid green image might be 'Success' or 'No confident detection found'
        # We just verify the pipeline executed end-to-end
        assert trace.execution_result is not None
        assert "status" in trace.execution_result

    def test_e2e_messy_query_2(self, agent, tmp_path):
        img_path = tmp_path / "test2.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="blue").save(img_path)
        
        trace = agent.route_and_configure(
            query="messy query 2: can you locate the built-up area and buildings?",
            image_paths=[str(img_path)]
        )
        assert trace.selected_task.value == "Text-Guided Region Grounding"
        assert trace.execution_result is not None

    def test_e2e_messy_query_3(self, agent, tmp_path):
        img_path = tmp_path / "test3.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="gray").save(img_path)
        
        trace = agent.route_and_configure(
            query="messy query 3: detect the road in this scene right now",
            image_paths=[str(img_path)]
        )
        assert trace.selected_task.value == "Text-Guided Region Grounding"
        assert trace.execution_result is not None
