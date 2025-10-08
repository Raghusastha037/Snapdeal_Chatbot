import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import json
from datetime import datetime
import re
from typing import List, Dict, Optional
from urllib.parse import urljoin, quote_plus
import random
from html.parser import HTMLParser

class SnapdealHTMLParser(HTMLParser):
    """Custom HTML parser to extract product information"""
    
    def __init__(self):
        super().__init__()
        self.products = []
        self.current_product = {}
        self.current_tag = None
        self.current_attrs = {}
        self.capturing = False
        self.capture_type = None
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        if tag == 'div':
            classes = self.current_attrs.get('class', '')
            if 'product-tuple' in classes or 'product' in classes:
                if self.current_product and any(self.current_product.values()):
                    self.products.append(self.current_product.copy())
                self.current_product = {}
        
        if tag in ['p', 'h2', 'h3', 'a']:
            classes = self.current_attrs.get('class', '')
            title_indicators = ['product-title', 'productTitle', 'prod-title', 'title']
            if any(indicator in classes for indicator in title_indicators):
                self.capturing = True
                self.capture_type = 'title'
        
        if tag == 'span':
            classes = self.current_attrs.get('class', '')
            price_indicators = ['product-price', 'price', 'selling-price', 'lfloat']
            if any(indicator in classes for indicator in price_indicators):
                self.capturing = True
                self.capture_type = 'price'
            
            discount_indicators = ['discount', 'product-discount', 'save']
            if any(indicator in classes for indicator in discount_indicators):
                self.capturing = True
                self.capture_type = 'discount'
            
            if 'strike' in classes or 'original' in classes:
                self.capturing = True
                self.capture_type = 'original_price'
        
        if tag == 'a' and 'href' in self.current_attrs:
            href = self.current_attrs['href']
            if '/product/' in href or 'dp/' in href:
                self.current_product['url'] = href
    
    def handle_data(self, data):
        if self.capturing and data.strip():
            data = data.strip()
            
            if self.capture_type == 'title':
                if 'title' not in self.current_product or len(data) > len(self.current_product.get('title', '')):
                    self.current_product['title'] = data
            
            elif self.capture_type == 'price':
                if '₹' in data or 'Rs' in data or data.replace(',', '').replace('.', '').isdigit():
                    self.current_product['price'] = data
            
            elif self.capture_type == 'discount':
                if '%' in data or 'off' in data.lower():
                    self.current_product['discount'] = data
            
            elif self.capture_type == 'original_price':
                if '₹' in data or 'Rs' in data:
                    self.current_product['original_price'] = data
    
    def handle_endtag(self, tag):
        if self.capturing:
            self.capturing = False
            self.capture_type = None
    
    def get_products(self):
        if self.current_product and any(self.current_product.values()):
            self.products.append(self.current_product.copy())
        return self.products


class AIAssistant:
    """Simple rule-based AI assistant for natural language understanding"""
    
    def __init__(self):
        self.intent_patterns = {
            'search_product': [
                r'show (?:me )?(.+)',
                r'find (?:me )?(.+)',
                r'search (?:for )?(.+)',
                r'looking for (.+)',
                r'i want (.+)',
                r'need (.+)',
                r'(.+) (?:under|below|less than) (?:rs |₹)?(\d+)',
                r'best (.+)',
                r'top (.+)',
                r'cheap(?:est)? (.+)',
                r'good (.+)',
            ],
            'price_query': [
                r'how much (?:is|are|does|do|cost) (.+)',
                r'price of (.+)',
                r'cost of (.+)',
                r'what(?:\'s| is) the price of (.+)',
            ],
            'policy_query': [
                r'(?:what(?:\'s| is))? (?:the )?(.+) policy',
                r'how (?:do|can) i (.+)',
                r'tell me about (.+)',
                r'information about (.+)',
                r'(?:explain|describe) (.+)',
            ],
            'comparison': [
                r'compare (.+) (?:and|vs|versus) (.+)',
                r'difference between (.+) and (.+)',
                r'which is better (.+) or (.+)',
            ],
            'recommendation': [
                r'recommend (?:me )?(.+)',
                r'suggest (?:me )?(.+)',
                r'what should i (?:buy|get) for (.+)',
            ],
            'greeting': [
                r'hi|hello|hey|greetings',
                r'good (?:morning|afternoon|evening)',
            ],
            'thanks': [
                r'thank(?:s| you)',
                r'appreciate it',
                r'helpful',
            ]
        }
        
        self.context_history = []
    
    def detect_intent(self, query: str) -> Dict[str, any]:
        """Detect user intent and extract entities"""
        query_lower = query.lower().strip()
        
        # Check for greetings
        for pattern in self.intent_patterns['greeting']:
            if re.search(pattern, query_lower):
                return {
                    'intent': 'greeting',
                    'query': query,
                    'entities': {}
                }
        
        # Check for thanks
        for pattern in self.intent_patterns['thanks']:
            if re.search(pattern, query_lower):
                return {
                    'intent': 'thanks',
                    'query': query,
                    'entities': {}
                }
        
        # Check for price queries
        for pattern in self.intent_patterns['price_query']:
            match = re.search(pattern, query_lower)
            if match:
                return {
                    'intent': 'price_query',
                    'query': match.group(1).strip(),
                    'entities': {'product': match.group(1).strip()}
                }
        
        # Check for policy queries
        for pattern in self.intent_patterns['policy_query']:
            match = re.search(pattern, query_lower)
            if match:
                return {
                    'intent': 'policy_query',
                    'query': match.group(1).strip(),
                    'entities': {'topic': match.group(1).strip()}
                }
        
        # Check for product search with price constraint
        price_pattern = r'(.+?)\s+(?:under|below|less than|within)\s+(?:rs |₹)?(\d+)'
        match = re.search(price_pattern, query_lower)
        if match:
            return {
                'intent': 'search_product',
                'query': match.group(1).strip(),
                'entities': {
                    'product': match.group(1).strip(),
                    'max_price': int(match.group(2))
                }
            }
        
        # Check for general product search
        for pattern in self.intent_patterns['search_product']:
            match = re.search(pattern, query_lower)
            if match:
                return {
                    'intent': 'search_product',
                    'query': match.group(1).strip(),
                    'entities': {'product': match.group(1).strip()}
                }
        
        # Default to search
        return {
            'intent': 'search_product',
            'query': query,
            'entities': {'product': query}
        }
    
    def enhance_query(self, intent_data: Dict) -> str:
        """Enhance query based on intent"""
        intent = intent_data['intent']
        query = intent_data['query']
        entities = intent_data['entities']
        
        if intent == 'search_product':
            # Don't lose the original query!
            enhanced = query
            
            # Add category keywords for better search
            category_keywords = {
                'phone': 'smartphone mobile',
                'laptop': 'laptop notebook computer',
                'shoe': 'shoes footwear',
                'dress': 'dress kurti clothing',
                'watch': 'watch smartwatch',
                'headphone': 'headphones earphones audio',
                'tablet': 'tablet ipad',
            }
            
            query_lower = query.lower()
            for key, value in category_keywords.items():
                if key in query_lower:
                    enhanced = f"{query} {value}"
                    break
            
            return enhanced
        
        return query
    
    def generate_conversational_response(self, intent_data: Dict, rag_response: str) -> str:
        """Generate more conversational response"""
        intent = intent_data['intent']
        
        if intent == 'greeting':
            return ("👋 Hello! Welcome to Snapdeal Shopping Assistant!\n\n"
                   "I can help you:\n"
                   "• Find products (smartphones, laptops, fashion, etc.)\n"
                   "• Compare prices and deals\n"
                   "• Answer questions about delivery, returns, and payments\n\n"
                   "What are you looking for today?")
        
        elif intent == 'thanks':
            return ("😊 You're welcome! Happy to help!\n\n"
                   "Feel free to ask if you need anything else.\n"
                   "Have a great shopping experience! 🛍️")
        
        elif intent == 'price_query':
            return f"💰 Price Information:\n\n{rag_response}"
        
        elif intent == 'policy_query':
            return f"📋 Policy Information:\n\n{rag_response}"
        
        elif intent == 'search_product':
            entities = intent_data['entities']
            header = "🛍️ Here are the products I found"
            
            if 'max_price' in entities:
                header += f" under ₹{entities['max_price']}"
            
            return f"{header}:\n\n{rag_response}"
        
        return rag_response


class SnapdealAPIClient:
    """Client to interact with Snapdeal's internal APIs"""
    
    def __init__(self):
        self.base_url = "https://www.snapdeal.com"
        self.api_base = "https://www.snapdeal.com/acors/json"
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        ]
    
    def _get_headers(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'application/json, text/javascript, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.snapdeal.com/',
            'X-Requested-With': 'XMLHttpRequest'
        }
    
    def search_products(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search products using Snapdeal's search API"""
        try:
            search_url = f"{self.api_base}/searchGoogleProducts/search/{quote_plus(query)}/0/20"
            response = requests.get(search_url, headers=self._get_headers(), timeout=15)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    products = []
                    
                    if isinstance(data, dict) and 'products' in data:
                        for item in data['products'][:max_results]:
                            products.append({
                                'title': item.get('title', item.get('name', '')),
                                'price': item.get('price', item.get('salePrice', '')),
                                'original_price': item.get('mrp', ''),
                                'discount': item.get('discount', ''),
                                'rating': item.get('rating', ''),
                                'url': item.get('url', item.get('link', ''))
                            })
                    
                    return products
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            print(f"  API search failed: {e}")
        
        return []


class SnapdealRAGChatbot:
    def __init__(self, pinecone_api_key: str):
        """Initialize the Snapdeal chatbot with AI assistant"""
        print("Initializing AI Assistant...")
        self.ai_assistant = AIAssistant()
        
        print("Initializing Pinecone...")
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "snapdeal-ai-assistant"
        
        print("Initializing TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=384,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1,
            sublinear_tf=True,
            norm='l2'
        )
        
        self.knowledge_base = []
        self.last_scrape_time = None
        self.scrape_interval = 1800
        self.vector_dimension = None  # Will be set after fitting vectorizer
        
        self.api_client = SnapdealAPIClient()
        self.base_url = "https://www.snapdeal.com"
        
        # First, fetch data and fit vectorizer to determine actual dimension
        self._prepare_knowledge_base()
        
        # Then setup Pinecone with correct dimension
        self._setup_pinecone_index()
        
        # Finally, index the data
        self._index_knowledge_base()
    
    def _prepare_knowledge_base(self):
        """Prepare knowledge base and determine vector dimension"""
        print("\n" + "="*80)
        print("🛒 SNAPDEAL LIVE DATA FETCHING".center(80))
        print("="*80 + "\n")
        
        self.knowledge_base = []
        
        search_queries = [
            'smartphones',
            'laptops',
            'mens shoes',
            'womens kurti',
            'headphones',
            'smartwatch',
            'tablet',
            'camera'
        ]
        
        print("📱 Fetching products via API...")
        
        for query in search_queries[:4]:
            print(f"  → Searching: {query}")
            products = self.api_client.search_products(query, max_results=10)
            
            for idx, product in enumerate(products):
                try:
                    title = product.get('title', '')
                    price = product.get('price', '')
                    
                    if not title or not price:
                        continue
                    
                    text = f"{title} - Price: {price}"
                    
                    if product.get('original_price'):
                        text += f" (MRP: {product['original_price']})"
                    
                    if product.get('discount'):
                        text += f" {product['discount']}"
                    
                    if product.get('rating'):
                        text += f". Rating: {product['rating']}"
                    
                    self.knowledge_base.append({
                        'id': f'product_{query}_{idx}',
                        'text': text,
                        'category': query,
                        'product_name': title,
                        'price': price,
                        'discount': product.get('discount', ''),
                        'product_url': product.get('url', ''),
                        'source': 'snapdeal_live',
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    continue
            
            print(f"    ✓ Added {len([p for p in self.knowledge_base if query in p['id']])} products")
            time.sleep(2)
        
        print("\n📌 Adding store information...")
        self.knowledge_base.extend(self._get_store_info())
        
        if len(self.knowledge_base) < 15:
            print("⚠ Adding fallback products...")
            self.knowledge_base.extend(self._get_fallback_products())
        
        print(f"\n✓ Total items collected: {len(self.knowledge_base)}")
        
        # Fit vectorizer to determine actual dimension
        all_texts = [doc['text'] for doc in self.knowledge_base]
        self.vectorizer.fit(all_texts)
        
        # Get actual vector dimension from fitted vectorizer
        sample_vec = self.vectorizer.transform([all_texts[0]]).toarray()[0]
        self.vector_dimension = len(sample_vec)
        
        print(f"✓ Vector dimension determined: {self.vector_dimension}")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        self.last_scrape_time = datetime.now()
    
    def _setup_pinecone_index(self):
        """Setup Pinecone index with correct dimension"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                print(f"Deleting old index: {self.index_name}")
                self.pc.delete_index(self.index_name)
                time.sleep(5)
            
            print(f"Creating new index: {self.index_name} (dimension: {self.vector_dimension})")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.vector_dimension,  # Use actual dimension from vectorizer
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            print("Waiting for index to be ready...")
            max_wait = 60
            waited = 0
            while waited < max_wait:
                try:
                    desc = self.pc.describe_index(self.index_name)
                    if desc.status.ready:
                        break
                except:
                    pass
                time.sleep(2)
                waited += 2
            
            self.index = self.pc.Index(self.index_name)
            print("✓ Index ready!")
            
        except Exception as e:
            print(f"Error setting up index: {e}")
            raise
    
    def _get_store_info(self) -> List[Dict]:
        """Store policies and information"""
        return [
            {
                'id': 'info_about',
                'text': 'Snapdeal is India\'s leading online shopping marketplace with millions of products across electronics, fashion, home, and more.',
                'category': 'info',
                'source': 'static',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'info_delivery',
                'text': 'Snapdeal offers Cash on Delivery (COD), free shipping on eligible products, and delivery within 2-7 business days across India.',
                'category': 'delivery',
                'source': 'static',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'info_return',
                'text': 'Easy returns within 7-30 days depending on product category. Return shipping is free. Full refund or replacement guaranteed.',
                'category': 'return_policy',
                'source': 'static',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'info_payment',
                'text': 'Payment options: Credit/Debit Cards, Net Banking, UPI (GPay, PhonePe, Paytm), Wallets, EMI options, and Cash on Delivery.',
                'category': 'payment',
                'source': 'static',
                'timestamp': datetime.now().isoformat()
            },
        ]
    
    def _get_fallback_products(self) -> List[Dict]:
        """Fallback products"""
        return [
            # Smartphones
            {
                'id': 'fb_mobile_1',
                'text': 'Samsung Galaxy M14 5G - Price: ₹12,990 (MRP: ₹16,990) 23% off. 6GB RAM, 128GB storage, 50MP camera. Rating: 4.3/5',
                'category': 'smartphones',
                'product_name': 'Samsung Galaxy M14 5G',
                'price': '₹12,990',
                'discount': '23% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_mobile_2',
                'text': 'Redmi 12 5G - Price: ₹10,999 (MRP: ₹14,999) 27% off. 4GB RAM, 128GB storage, 50MP camera. Rating: 4.1/5',
                'category': 'smartphones',
                'product_name': 'Redmi 12 5G',
                'price': '₹10,999',
                'discount': '27% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_mobile_3',
                'text': 'Realme Narzo 60 5G - Price: ₹17,999 (MRP: ₹24,999) 28% off. 8GB RAM, 128GB storage, 64MP camera. Rating: 4.4/5',
                'category': 'smartphones',
                'product_name': 'Realme Narzo 60 5G',
                'price': '₹17,999',
                'discount': '28% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_mobile_4',
                'text': 'Samsung Galaxy A14 5G - Price: ₹14,490 (MRP: ₹20,990) 31% off. 6GB RAM, 128GB storage, 50MP camera. Rating: 4.2/5',
                'category': 'smartphones',
                'product_name': 'Samsung Galaxy A14 5G',
                'price': '₹14,490',
                'discount': '31% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            # Laptops
            {
                'id': 'fb_laptop_1',
                'text': 'HP 14s Laptop - Price: ₹32,990 (MRP: ₹45,000) 27% off. Intel Core i3, 8GB RAM, 512GB SSD, Windows 11. Rating: 4.2/5',
                'category': 'laptops',
                'product_name': 'HP 14s Laptop',
                'price': '₹32,990',
                'discount': '27% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_laptop_2',
                'text': 'Lenovo IdeaPad Slim 3 - Price: ₹29,990 (MRP: ₹42,000) 29% off. Intel Celeron, 8GB RAM, 256GB SSD, Windows 11. Rating: 4.0/5',
                'category': 'laptops',
                'product_name': 'Lenovo IdeaPad Slim 3',
                'price': '₹29,990',
                'discount': '29% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_laptop_3',
                'text': 'Dell Vostro 3420 - Price: ₹38,990 (MRP: ₹52,000) 25% off. Intel Core i3, 8GB RAM, 512GB SSD, Windows 11 Pro. Rating: 4.3/5',
                'category': 'laptops',
                'product_name': 'Dell Vostro 3420',
                'price': '₹38,990',
                'discount': '25% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_laptop_4',
                'text': 'ASUS Vivobook 15 - Price: ₹35,990 (MRP: ₹48,000) 25% off. Intel Core i3, 8GB RAM, 512GB SSD, Windows 11. Rating: 4.1/5',
                'category': 'laptops',
                'product_name': 'ASUS Vivobook 15',
                'price': '₹35,990',
                'discount': '25% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            # Headphones
            {
                'id': 'fb_headphone_1',
                'text': 'boAt Rockerz 450 - Price: ₹1,299 (MRP: ₹2,990) 57% off. Bluetooth headphones, 15hr battery, bass boost. Rating: 4.2/5',
                'category': 'headphones',
                'product_name': 'boAt Rockerz 450',
                'price': '₹1,299',
                'discount': '57% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_headphone_2',
                'text': 'Noise Buds VS104 - Price: ₹999 (MRP: ₹2,499) 60% off. TWS earbuds, 30hr playtime, IPX5 waterproof. Rating: 4.0/5',
                'category': 'headphones',
                'product_name': 'Noise Buds VS104',
                'price': '₹999',
                'discount': '60% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
        ]
    
    def _index_knowledge_base(self):
        """Index knowledge base into Pinecone"""
        print("\nIndexing knowledge base...")
        
        all_texts = [doc['text'] for doc in self.knowledge_base]
        embeddings = self.vectorizer.transform(all_texts).toarray()
        
        vectors = []
        for i, doc in enumerate(self.knowledge_base):
            embedding = embeddings[i]
            
            if np.sum(np.abs(embedding)) < 1e-10:
                embedding = np.random.randn(self.vector_dimension) * 0.01
            
            metadata = {
                'text': doc['text'],
                'category': doc['category'],
                'source': doc.get('source', 'unknown'),
                'timestamp': doc.get('timestamp', datetime.now().isoformat())
            }
            
            if 'product_name' in doc:
                metadata['product_name'] = doc['product_name'][:500]
            if 'price' in doc:
                metadata['price'] = doc['price']
            if 'discount' in doc:
                metadata['discount'] = doc['discount']
            if 'product_url' in doc:
                metadata['product_url'] = doc['product_url']
            
            vectors.append({
                'id': doc['id'],
                'values': embedding.tolist(),
                'metadata': metadata
            })
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        time.sleep(3)
        print(f"✓ Indexed {len(vectors)} items!")
        
        print("\n" + "="*80)
        print("✅ DATA INDEXING COMPLETE".center(80))
        print("="*80 + "\n")
    
    def retrieve_relevant_info(self, query: str, top_k: int = 5, max_price: Optional[int] = None):
        """Retrieve relevant information with optional price filtering"""
        query_vec = self.vectorizer.transform([query]).toarray()[0]
        
        # Always try keyword search first for better results
        keyword_results = self._keyword_search(query, top_k, max_price)
        
        if np.sum(np.abs(query_vec)) < 1e-10:
            print("  [Debug] Using keyword search (zero query vector)")
            return keyword_results
        
        try:
            results = self.index.query(
                vector=query_vec.tolist(),
                top_k=top_k * 3 if max_price else top_k,
                include_metadata=True
            )
            
            matches = results.get('matches', [])
            
            if not matches:
                print("  [Debug] No Pinecone matches, using keyword search")
                return keyword_results
            
            # Convert Pinecone format to consistent format with 'score' key
            formatted_matches = []
            for match in matches:
                formatted_matches.append({
                    'id': match.get('id', ''),
                    'score': match.get('score', 0.0),
                    'metadata': match.get('metadata', {})
                })
            
            # Filter by price if specified
            if max_price:
                filtered_matches = []
                for match in formatted_matches:
                    price_str = match['metadata'].get('price', '')
                    try:
                        price_match = re.search(r'₹?\s*([0-9,]+)', price_str)
                        if price_match:
                            price = int(price_match.group(1).replace(',', ''))
                            if price <= max_price:
                                filtered_matches.append(match)
                    except:
                        continue
                
                if filtered_matches:
                    return filtered_matches[:top_k]
            
            # If Pinecone results have low scores, prefer keyword search
            if formatted_matches and formatted_matches[0]['score'] < 0.3:
                if keyword_results and keyword_results[0]['score'] > 0:
                    print("  [Debug] Keyword search has better results")
                    return keyword_results
            
            return formatted_matches[:top_k] if formatted_matches else keyword_results
            
        except Exception as e:
            print(f"  [Debug] Pinecone query error: {e}")
            return keyword_results
    
    def _keyword_search(self, query: str, top_k: int = 5, max_price: Optional[int] = None):
        """Keyword-based fallback search with fuzzy matching"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        print(f"  [Debug] Keyword search: '{query}' -> words: {query_words}")
        
        scores = []
        for doc in self.knowledge_base:
            # Price filtering
            if max_price:
                price_str = doc.get('price', '')
                try:
                    price_match = re.search(r'₹?\s*([0-9,]+)', price_str)
                    if price_match:
                        price = int(price_match.group(1).replace(',', ''))
                        if price > max_price:
                            continue
                except:
                    pass
            
            text_lower = doc['text'].lower()
            text_words = set(text_lower.split())
            
            # Calculate overlap score
            overlap = len(query_words.intersection(text_words))
            
            # Boost for category match
            category = doc.get('category', '').lower()
            for qword in query_words:
                if qword in category or category in qword:
                    overlap += 3
                    break
            
            # Boost for product name match
            if 'product_name' in doc:
                product_name_lower = doc['product_name'].lower()
                for word in query_words:
                    if len(word) > 2:  # Skip very short words
                        if word in product_name_lower:
                            overlap += 4
                        # Partial match
                        elif any(word in pword or pword in word for pword in product_name_lower.split()):
                            overlap += 2
            
            # Fuzzy match for common typos
            if 'smasung' in query_lower or 'samung' in query_lower:
                if 'samsung' in text_lower:
                    overlap += 5
            
            # Check for phone/smartphone synonyms
            phone_keywords = {'phone', 'smartphone', 'mobile'}
            if phone_keywords.intersection(query_words):
                if phone_keywords.intersection(text_words) or 'smartphone' in category:
                    overlap += 2
            
            # Check for laptop keywords
            laptop_keywords = {'laptop', 'notebook', 'computer'}
            if laptop_keywords.intersection(query_words):
                if laptop_keywords.intersection(text_words) or 'laptop' in category:
                    overlap += 2
            
            if overlap > 0:
                score_val = overlap / max(len(query_words), 1)
                scores.append({
                    'id': doc['id'],
                    'score': score_val,
                    'metadata': doc
                })
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        result = scores[:top_k]
        if result:
            print(f"  [Debug] Keyword search found {len(result)} results, top score: {result[0]['score']:.2f}")
        else:
            print(f"  [Debug] Keyword search found 0 results from {len(self.knowledge_base)} docs")
        
        return result
    
    def generate_response(self, query: str, retrieved_docs: list) -> str:
        """Generate response"""
        if not retrieved_docs:
            return ("No specific products found. Try different keywords or ask about:\n"
                   "• 'smartphones under 15000'\n"
                   "• 'best laptops for students'\n"
                   "• 'delivery policy'\n")
        
        # Check if we have valid results
        has_valid_results = False
        for doc in retrieved_docs:
            if doc.get('score', 0) > 0.01:
                has_valid_results = True
                break
        
        if not has_valid_results:
            return ("No specific products found. Try different keywords or ask about:\n"
                   "• 'smartphones under 15000'\n"
                   "• 'best laptops for students'\n"
                   "• 'delivery policy'\n")
        
        response = []
        
        for i, doc in enumerate(retrieved_docs[:5], 1):  # Limit to top 5
            text = doc['metadata'].get('text', '')
            score = doc.get('score', 0)
            
            if not text or score <= 0.01:
                continue
            
            response.append(f"{i}. {text}")
            
            # Add product URL if available
            product_url = doc['metadata'].get('product_url', '')
            if product_url and product_url.startswith('http'):
                response.append(f"   🔗 {product_url}")
            
            # Add freshness info
            timestamp = doc['metadata'].get('timestamp', '')
            if timestamp:
                try:
                    ts = datetime.fromisoformat(timestamp)
                    age = datetime.now() - ts
                    if age.seconds < 3600:
                        freshness = f"{age.seconds//60}m ago"
                    elif age.days == 0:
                        freshness = f"{age.seconds//3600}h ago"
                    else:
                        freshness = f"{age.days}d ago"
                    
                    source = doc['metadata'].get('source', 'live')
                    response.append(f"   📊 [{source} | {freshness}]\n")
                except:
                    response.append("")
        
        if not response:
            return ("No specific products found. Try different keywords or ask about:\n"
                   "• 'smartphones under 15000'\n"
                   "• 'best laptops for students'\n"
                   "• 'delivery policy'\n")
        
        return '\n'.join(response)
    
    def chat(self, user_query: str) -> str:
        """Main chat method with AI assistant"""
        intent_data = self.ai_assistant.detect_intent(user_query)
        
        if intent_data['intent'] in ['greeting', 'thanks']:
            return self.ai_assistant.generate_conversational_response(intent_data, "")
        
        enhanced_query = self.ai_assistant.enhance_query(intent_data)
        max_price = intent_data['entities'].get('max_price')
        
        # Debug info
        print(f"  [Debug] Intent: {intent_data['intent']}")
        print(f"  [Debug] Enhanced query: '{enhanced_query}'")
        if max_price:
            print(f"  [Debug] Max price filter: ₹{max_price}")
        
        retrieved_docs = self.retrieve_relevant_info(enhanced_query, top_k=5, max_price=max_price)
        
        print(f"  [Debug] Retrieved {len(retrieved_docs)} documents")
        if retrieved_docs:
            print(f"  [Debug] Top score: {retrieved_docs[0].get('score', 0):.4f}")
        
        rag_response = self.generate_response(enhanced_query, retrieved_docs)
        final_response = self.ai_assistant.generate_conversational_response(intent_data, rag_response)
        
        return final_response


def main():
    print("=" * 80)
    print("🤖 SNAPDEAL AI SHOPPING ASSISTANT".center(80))
    print("Powered by RAG + Natural Language Understanding".center(80))
    print("=" * 80)
    
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    if not pinecone_api_key:
        pinecone_api_key = input("\nEnter Pinecone API key: ").strip()
    
    if not pinecone_api_key:
        print("❌ Pinecone API key required!")
        return
    
    try:
        print("\n⏳ Initializing AI Assistant and fetching live data...")
        print("This may take 30-60 seconds...\n")
        
        chatbot = SnapdealRAGChatbot(pinecone_api_key=pinecone_api_key)
        
        print("\n" + "=" * 80)
        print("✅ AI ASSISTANT READY!".center(80))
        print("=" * 80)
        
        print("\n🤖 I'm your intelligent shopping assistant!")
        print("\n💡 Try natural language queries like:")
        print("  • 'Hi, I'm looking for a good smartphone'")
        print("  • 'Show me laptops under 30000'")
        print("  • 'What's the delivery policy?'")
        print("  • 'Find me cheap headphones'")
        print("  • 'How much does the Samsung phone cost?'")
        
        print("\n⚙️  Commands: 'refresh' (update data) | 'quit' (exit)")
        print("\n" + "=" * 80 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 Thank you for using Snapdeal AI Assistant!")
                print("Happy Shopping! 🛍️\n")
                break
            
            if user_input.lower() == 'refresh':
                print("\n🔄 Refreshing product database...")
                chatbot._prepare_knowledge_base()
                chatbot._setup_pinecone_index()
                chatbot._index_knowledge_base()
                print("✅ Database refreshed!\n")
                continue
            
            if not user_input:
                continue
            
            try:
                response = chatbot.chat(user_input)
                print(f"\n🤖 Assistant:\n{response}\n")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try rephrasing your question.\n")
    
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Verify your Pinecone API key is valid")
        print("  2. Check your internet connection")
        print("  3. Install required packages:")
        print("     pip install pinecone-client scikit-learn numpy requests")
        print("  4. Ensure you have pinecone-client>=3.0.0")


if __name__ == "__main__":
    main()