"""
query_expansion

Strategy B — AI-enhanced retrieval via generative query expansion.

The expander takes a user question and uses a GenerativeModel to produce
``n`` semantically equivalent (or complementary) reformulations.  All
expansions are embedded and searched independently; the union of results is
then re-ranked by maximum score across all variants.

Mock mode:

When ``mock=True`` the module replaces the Vertex AI GenerativeModel with
``tests.mocks.gcp_mocks.MockGenerativeModel``, which returns
deterministic rewrites without touching the network.  This keeps the CI
pipeline fast and credentials-free.

Production mode:

Set ``mock=False`` and ensure the service account has
``roles/aiplatform.user``.  The module will call
``vertexai.generative_models.GenerativeModel`` (Gemini) for real.
"""

from __future__ import annotations

import re




class GenerativeQueryExpander:
    """
    Rewrites a user query into ``n_expansions`` alternative phrasings.
    """

    _SYSTEM_PROMPT = (
        """You are an expert at query reformulation for information retrieval. 
        Given a user question, generate {n} semantically equivalent but 
        differently phrased versions that cover different facets of the 
        same information need.  Output ONLY the questions, one per line, 
        with no numbering, bullets, or extra commentary."""
    )

    def __init__(
            self,
            n_expansions: int = 3,
            mock: bool = True,
            temperature: float = 0.3,
        ) -> None:
            self.n_expansions = n_expansions
            self.mock = mock
            self.temperature = temperature
            self._model = self._build_model()

    def _build_model(self):

        if self.mock:
            from tests.mocks.gcp_mocks import MockGenerativeModel
            return MockGenerativeModel()

        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(
            project=self.project, # type: ignore
            location=self.location, # type: ignore
        )

        return GenerativeModel("gemini-1.5-flash")
    

    def expand(self, query: str) -> list[str]:
        prompt = (
            self._SYSTEM_PROMPT.format(n = self.n_expansions) + f"\n\nOriginal Question: {query}"
        )
        response = self._model.generate_content(prompt)

        return self._parse_response(response.text)
    
    def _parse_response(self, raw: str) -> list[str]:
        """Split model output into individual query strings."""
        lines = [
            re.sub(r"^\s*[\d\.\-\*]+\s*", "", ln).strip()
            for ln in raw.splitlines()
            if ln.strip()
        ]
        return [ln for ln in lines if ln][: self.n_expansions]