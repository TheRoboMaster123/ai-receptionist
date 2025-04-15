from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sqlalchemy.orm import Session

class KnowledgeBaseManager:
    def __init__(self, db: Session):
        self.db = db
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_dir = Path("knowledge_base_cache")
        self.cache_dir.mkdir(exist_ok=True)

    async def add_document(
        self,
        business_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_type: str = "general"
    ) -> Dict[str, Any]:
        """Add a document to the knowledge base"""
        # Generate embeddings
        embeddings = self.model.encode([content])[0]
        
        # Create document entry
        doc = {
            "content": content,
            "embeddings": embeddings.tolist(),
            "metadata": metadata or {},
            "type": doc_type,
            "added_at": datetime.utcnow().isoformat()
        }
        
        # Save to business knowledge base
        kb_path = self.cache_dir / f"{business_id}_kb.json"
        
        try:
            if kb_path.exists():
                kb = json.loads(kb_path.read_text())
                kb["documents"].append(doc)
            else:
                kb = {"documents": [doc]}
            
            kb_path.write_text(json.dumps(kb, indent=2))
            return doc
        except Exception as e:
            raise Exception(f"Error adding document: {str(e)}")

    async def search_documents(
        self,
        business_id: str,
        query: str,
        top_k: int = 3,
        threshold: float = 0.6,
        doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search knowledge base documents by similarity"""
        kb_path = self.cache_dir / f"{business_id}_kb.json"
        
        if not kb_path.exists():
            return []
        
        try:
            # Load knowledge base
            kb = json.loads(kb_path.read_text())
            
            # Filter by doc_type if specified
            docs = kb["documents"]
            if doc_type:
                docs = [d for d in docs if d["type"] == doc_type]
            
            if not docs:
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            doc_embeddings = np.array([d["embeddings"] for d in docs])
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            # Get top matches above threshold
            matches = []
            for idx, score in enumerate(similarities):
                if score >= threshold:
                    doc = docs[idx].copy()
                    doc["similarity_score"] = float(score)
                    matches.append(doc)
            
            # Sort by similarity score
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            return matches[:top_k]
            
        except Exception as e:
            raise Exception(f"Error searching documents: {str(e)}")

    async def delete_document(
        self,
        business_id: str,
        doc_id: str
    ) -> bool:
        """Delete a document from the knowledge base"""
        kb_path = self.cache_dir / f"{business_id}_kb.json"
        
        if not kb_path.exists():
            return False
        
        try:
            kb = json.loads(kb_path.read_text())
            kb["documents"] = [
                d for d in kb["documents"]
                if d.get("metadata", {}).get("id") != doc_id
            ]
            kb_path.write_text(json.dumps(kb, indent=2))
            return True
        except Exception:
            return False

    async def get_documents(
        self,
        business_id: str,
        doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all documents for a business"""
        kb_path = self.cache_dir / f"{business_id}_kb.json"
        
        if not kb_path.exists():
            return []
        
        try:
            kb = json.loads(kb_path.read_text())
            docs = kb["documents"]
            
            if doc_type:
                docs = [d for d in docs if d["type"] == doc_type]
            
            return docs
        except Exception:
            return [] 