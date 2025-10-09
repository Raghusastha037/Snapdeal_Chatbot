import os
from pinecone import Pinecone, ServerlessSpec
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
        self.vector_dimension = None
        
        self.api_client = SnapdealAPIClient()
        self.base_url = "https://www.snapdeal.com"
        
        # Prepare knowledge base and fit vectorizer
        self._prepare_knowledge_base()
        
        # Setup Pinecone with correct dimension
        self._setup_pinecone_index()
        
        # Index the data
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
        if self.knowledge_base:
            all_texts = [doc['text'] for doc in self.knowledge_base]
            self.vectorizer.fit(all_texts)
            
            # Get actual vector dimension
            sample_vec = self.vectorizer.transform([all_texts[0]]).toarray()[0]
            self.vector_dimension = len(sample_vec)
            
            print(f"✓ Vector dimension determined: {self.vector_dimension}")
            print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        else:
            raise ValueError("No data in knowledge base to determine vector dimension")
        
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
                dimension=self.vector_dimension,
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
            {
                'id': 'fb_mobile_5',
                'text': 'OnePlus Nord CE 3 Lite 5G - Price: ₹17,499 (MRP: ₹22,999) 24% off. 8GB RAM, 128GB storage, 108MP camera. Rating: 4.4/5',
                'category': 'smartphones',
                'product_name': 'OnePlus Nord CE 3 Lite 5G',
                'price': '₹17,499',
                'discount': '24% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_mobile_6',
                'text': 'iQOO Z9 5G - Price: ₹18,999 (MRP: ₹23,999) 21% off. 8GB RAM, 256GB storage, 50MP Sony IMX camera. Rating: 4.3/5',
                'category': 'smartphones',
                'product_name': 'iQOO Z9 5G',
                'price': '₹18,999',
                'discount': '21% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_mobile_7',
                'text': 'Motorola G54 5G - Price: ₹13,999 (MRP: ₹18,999) 26% off. 12GB RAM, 256GB storage, 50MP OIS camera. Rating: 4.2/5',
                'category': 'smartphones',
                'product_name': 'Motorola G54 5G',
                'price': '₹13,999',
                'discount': '26% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_mobile_8',
                'text': 'Poco X6 5G - Price: ₹19,999 (MRP: ₹26,999) 26% off. 8GB RAM, 256GB storage, 64MP camera, 120Hz display. Rating: 4.5/5',
                'category': 'smartphones',
                'product_name': 'Poco X6 5G',
                'price': '₹19,999',
                'discount': '26% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_mobile_9',
                'text': 'Vivo T2x 5G - Price: ₹12,499 (MRP: ₹16,999) 26% off. 6GB RAM, 128GB storage, 50MP camera. Rating: 4.0/5',
                'category': 'smartphones',
                'product_name': 'Vivo T2x 5G',
                'price': '₹12,499',
                'discount': '26% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_mobile_10',
                'text': 'Oppo A78 5G - Price: ₹16,999 (MRP: ₹21,999) 23% off. 8GB RAM, 128GB storage, 50MP camera. Rating: 4.1/5',
                'category': 'smartphones',
                'product_name': 'Oppo A78 5G',
                'price': '₹16,999',
                'discount': '23% off',
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
            {
                'id': 'fb_laptop_5',
                'text': 'Acer Aspire 3 - Price: ₹31,499 (MRP: ₹44,990) 30% off. AMD Ryzen 3, 8GB RAM, 512GB SSD, Windows 11. Rating: 4.1/5',
                'category': 'laptops',
                'product_name': 'Acer Aspire 3',
                'price': '₹31,499',
                'discount': '30% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_laptop_6',
                'text': 'HP Pavilion 15 - Price: ₹52,990 (MRP: ₹68,000) 22% off. Intel Core i5, 16GB RAM, 512GB SSD, Windows 11. Rating: 4.4/5',
                'category': 'laptops',
                'product_name': 'HP Pavilion 15',
                'price': '₹52,990',
                'discount': '22% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_laptop_7',
                'text': 'ASUS TUF Gaming F15 - Price: ₹67,990 (MRP: ₹89,990) 24% off. Intel i5, RTX 3050 GPU, 16GB RAM, 512GB SSD. Rating: 4.5/5',
                'category': 'laptops',
                'product_name': 'ASUS TUF Gaming F15',
                'price': '₹67,990',
                'discount': '24% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_laptop_8',
                'text': 'MSI GF63 Thin - Price: ₹54,990 (MRP: ₹72,990) 25% off. Intel Core i5, GTX 1650, 8GB RAM, 512GB SSD. Rating: 4.3/5',
                'category': 'laptops',
                'product_name': 'MSI GF63 Thin',
                'price': '₹54,990',
                'discount': '25% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_laptop_9',
                'text': 'Lenovo Yoga Slim 7 - Price: ₹59,990 (MRP: ₹79,990) 25% off. AMD Ryzen 5, 16GB RAM, 512GB SSD, 14" FHD. Rating: 4.5/5',
                'category': 'laptops',
                'product_name': 'Lenovo Yoga Slim 7',
                'price': '₹59,990',
                'discount': '25% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_laptop_10',
                'text': 'Apple MacBook Air M1 - Price: ₹74,990 (MRP: ₹92,900) 19% off. Apple M1 chip, 8GB RAM, 256GB SSD, 13.3" Retina. Rating: 4.8/5',
                'category': 'laptops',
                'product_name': 'Apple MacBook Air M1',
                'price': '₹74,990',
                'discount': '19% off',
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
            {
                'id': 'fb_headphone_3',
                'text': 'JBL Tune 510BT - Price: ₹2,999 (MRP: ₹4,999) 40% off. Wireless Bluetooth headphones, 40hr battery, deep bass. Rating: 4.3/5',
                'category': 'headphones',
                'product_name': 'JBL Tune 510BT',
                'price': '₹2,999',
                'discount': '40% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_headphone_4',
                'text': 'Sony WH-CH520 - Price: ₹4,490 (MRP: ₹6,990) 36% off. On-ear wireless headphones, 50hr battery, mic support. Rating: 4.5/5',
                'category': 'headphones',
                'product_name': 'Sony WH-CH520',
                'price': '₹4,490',
                'discount': '36% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_headphone_5',
                'text': 'OnePlus Buds Z2 - Price: ₹3,499 (MRP: ₹4,999) 30% off. TWS earbuds, ANC, 38hr battery, fast charging. Rating: 4.3/5',
                'category': 'headphones',
                'product_name': 'OnePlus Buds Z2',
                'price': '₹3,499',
                'discount': '30% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_headphone_6',
                'text': 'Realme Buds Air 3 - Price: ₹2,799 (MRP: ₹4,999) 44% off. TWS earbuds, ANC, 30hr playtime, low latency. Rating: 4.2/5',
                'category': 'headphones',
                'product_name': 'Realme Buds Air 3',
                'price': '₹2,799',
                'discount': '44% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Smartwatches
            {
                'id': 'fb_watch_1',
                'text': 'Noise ColorFit Icon 2 - Price: ₹1,799 (MRP: ₹4,999) 64% off. 1.8" AMOLED display, BT calling, 10-day battery. Rating: 4.1/5',
                'category': 'smartwatch',
                'product_name': 'Noise ColorFit Icon 2',
                'price': '₹1,799',
                'discount': '64% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_watch_2',
                'text': 'Fire-Boltt Ninja Call Pro Plus - Price: ₹1,499 (MRP: ₹6,999) 79% off. 1.83" HD display, Bluetooth calling, heart rate. Rating: 4.2/5',
                'category': 'smartwatch',
                'product_name': 'Fire-Boltt Ninja Call Pro Plus',
                'price': '₹1,499',
                'discount': '79% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_watch_3',
                'text': 'boAt Wave Call - Price: ₹1,999 (MRP: ₹5,990) 67% off. 1.83" display, BT calling, 7-day battery, IP68. Rating: 4.0/5',
                'category': 'smartwatch',
                'product_name': 'boAt Wave Call',
                'price': '₹1,999',
                'discount': '67% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_watch_4',
                'text': 'Amazfit Bip 3 Pro - Price: ₹3,499 (MRP: ₹5,999) 42% off. 1.69" display, GPS, 14-day battery, 60+ sports modes. Rating: 4.3/5',
                'category': 'smartwatch',
                'product_name': 'Amazfit Bip 3 Pro',
                'price': '₹3,499',
                'discount': '42% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_watch_5',
                'text': 'Titan Smart Pro - Price: ₹5,495 (MRP: ₹7,995) 31% off. 1.96" AMOLED, BT calling, premium design. Rating: 4.4/5',
                'category': 'smartwatch',
                'product_name': 'Titan Smart Pro',
                'price': '₹5,495',
                'discount': '31% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Televisions
            {
                'id': 'fb_tv_1',
                'text': 'Mi 43-inch Smart TV 5A - Price: ₹23,999 (MRP: ₹31,999) 25% off. Full HD LED, Android TV, Dolby Audio. Rating: 4.3/5',
                'category': 'television',
                'product_name': 'Mi Smart TV 5A 43-inch',
                'price': '₹23,999',
                'discount': '25% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_tv_2',
                'text': 'Samsung Crystal 4K UHD 43-inch - Price: ₹28,990 (MRP: ₹38,990) 26% off. 4K Ultra HD, HDR10+, Alexa built-in. Rating: 4.4/5',
                'category': 'television',
                'product_name': 'Samsung Crystal 4K UHD 43-inch',
                'price': '₹28,990',
                'discount': '26% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_tv_3',
                'text': 'LG 32-inch HD Ready LED TV - Price: ₹14,990 (MRP: ₹19,990) 25% off. HD Ready, 60Hz, virtual surround sound. Rating: 4.2/5',
                'category': 'television',
                'product_name': 'LG 32-inch HD Ready LED TV',
                'price': '₹14,990',
                'discount': '25% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_tv_4',
                'text': 'Sony Bravia 55-inch 4K UHD - Price: ₹54,990 (MRP: ₹74,990) 27% off. 4K HDR, Android TV, Google Assistant. Rating: 4.6/5',
                'category': 'television',
                'product_name': 'Sony Bravia 55-inch 4K UHD',
                'price': '₹54,990',
                'discount': '27% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_tv_5',
                'text': 'OnePlus Y1S Pro 50-inch 4K - Price: ₹31,999 (MRP: ₹42,999) 26% off. 4K UHD, Android TV 11, Dolby Audio. Rating: 4.3/5',
                'category': 'television',
                'product_name': 'OnePlus Y1S Pro 50-inch',
                'price': '₹31,999',
                'discount': '26% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Home Appliances
            {
                'id': 'fb_home_1',
                'text': 'LG 7kg Washing Machine - Price: ₹18,490 (MRP: ₹24,990) 26% off. Front load, inverter motor, energy efficient. Rating: 4.4/5',
                'category': 'home_appliances',
                'product_name': 'LG 7kg Washing Machine',
                'price': '₹18,490',
                'discount': '26% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_home_2',
                'text': 'Samsung 253L Refrigerator - Price: ₹23,990 (MRP: ₹29,990) 20% off. Digital inverter, 3-star rating, double door. Rating: 4.5/5',
                'category': 'home_appliances',
                'product_name': 'Samsung 253L Refrigerator',
                'price': '₹23,990',
                'discount': '20% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_home_3',
                'text': 'Whirlpool 1.5 Ton AC - Price: ₹29,990 (MRP: ₹42,990) 30% off. 3-star, split AC, copper condenser, 6th sense. Rating: 4.3/5',
                'category': 'home_appliances',
                'product_name': 'Whirlpool 1.5 Ton AC',
                'price': '₹29,990',
                'discount': '30% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_home_4',
                'text': 'IFB 25L Convection Microwave - Price: ₹12,990 (MRP: ₹17,990) 28% off. 25L capacity, auto cook menu, child lock. Rating: 4.4/5',
                'category': 'home_appliances',
                'product_name': 'IFB 25L Convection Microwave',
                'price': '₹12,990',
                'discount': '28% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_home_5',
                'text': 'Philips Air Fryer - Price: ₹7,999 (MRP: ₹12,995) 38% off. 4.1L capacity, rapid air technology, timer. Rating: 4.5/5',
                'category': 'home_appliances',
                'product_name': 'Philips Air Fryer',
                'price': '₹7,999',
                'discount': '38% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Shoes
            {
                'id': 'fb_shoes_1',
                'text': 'Nike Revolution 6 - Price: ₹3,295 (MRP: ₹4,995) 34% off. Running shoes, lightweight, breathable mesh. Rating: 4.3/5',
                'category': 'shoes',
                'product_name': 'Nike Revolution 6',
                'price': '₹3,295',
                'discount': '34% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_shoes_2',
                'text': 'Adidas Ultraboost 22 - Price: ₹8,999 (MRP: ₹16,999) 47% off. Premium running shoes, boost cushioning. Rating: 4.6/5',
                'category': 'shoes',
                'product_name': 'Adidas Ultraboost 22',
                'price': '₹8,999',
                'discount': '47% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_shoes_3',
                'text': 'Puma Softride Enzo - Price: ₹2,799 (MRP: ₹5,999) 53% off. Sports shoes, SoftFoam+ insole, casual wear. Rating: 4.2/5',
                'category': 'shoes',
                'product_name': 'Puma Softride Enzo',
                'price': '₹2,799',
                'discount': '53% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_shoes_4',
                'text': 'Reebok Energen Plus - Price: ₹2,199 (MRP: ₹4,999) 56% off. Running shoes, FuelFoam midsole, durable. Rating: 4.1/5',
                'category': 'shoes',
                'product_name': 'Reebok Energen Plus',
                'price': '₹2,199',
                'discount': '56% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_shoes_5',
                'text': 'Campus North Plus - Price: ₹999 (MRP: ₹1,999) 50% off. Casual shoes, memory foam, all-day comfort. Rating: 4.0/5',
                'category': 'shoes',
                'product_name': 'Campus North Plus',
                'price': '₹999',
                'discount': '50% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Fashion - Men
            {
                'id': 'fb_fashion_men_1',
                'text': 'Levi\'s Men Slim Fit Jeans - Price: ₹1,799 (MRP: ₹2,999) 40% off. Blue denim, stretch fabric, mid-rise. Rating: 4.4/5',
                'category': 'mens_fashion',
                'product_name': 'Levi\'s Men Slim Fit Jeans',
                'price': '₹1,799',
                'discount': '40% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_fashion_men_2',
                'text': 'Allen Solly Men Formal Shirt - Price: ₹899 (MRP: ₹1,999) 55% off. Cotton blend, regular fit, blue checks. Rating: 4.2/5',
                'category': 'mens_fashion',
                'product_name': 'Allen Solly Men Formal Shirt',
                'price': '₹899',
                'discount': '55% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_fashion_men_3',
                'text': 'Peter England Blazer - Price: ₹2,499 (MRP: ₹4,999) 50% off. Formal blazer, slim fit, navy blue. Rating: 4.3/5',
                'category': 'mens_fashion',
                'product_name': 'Peter England Blazer',
                'price': '₹2,499',
                'discount': '50% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_fashion_men_4',
                'text': 'US Polo T-Shirt - Price: ₹699 (MRP: ₹1,599) 56% off. Cotton polo t-shirt, casual wear, multiple colors. Rating: 4.1/5',
                'category': 'mens_fashion',
                'product_name': 'US Polo T-Shirt',
                'price': '₹699',
                'discount': '56% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_fashion_men_5',
                'text': 'Raymond Men Trousers - Price: ₹1,299 (MRP: ₹2,999) 57% off. Formal trousers, flat front, wrinkle-free. Rating: 4.4/5',
                'category': 'mens_fashion',
                'product_name': 'Raymond Men Trousers',
                'price': '₹1,299',
                'discount': '57% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Fashion - Women
            {
                'id': 'fb_fashion_women_1',
                'text': 'W Women Kurta Set - Price: ₹899 (MRP: ₹2,499) 64% off. Cotton kurta with palazzo, floral print, casual. Rating: 4.3/5',
                'category': 'womens_fashion',
                'product_name': 'W Women Kurta Set',
                'price': '₹899',
                'discount': '64% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_fashion_women_2',
                'text': 'Libas Ethnic Dress - Price: ₹1,199 (MRP: ₹3,999) 70% off. Anarkali dress, embroidered, party wear. Rating: 4.5/5',
                'category': 'womens_fashion',
                'product_name': 'Libas Ethnic Dress',
                'price': '₹1,199',
                'discount': '70% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_fashion_women_3',
                'text': 'Biba Women Salwar Suit - Price: ₹1,799 (MRP: ₹4,999) 64% off. Cotton salwar suit, printed, dupatta included. Rating: 4.4/5',
                'category': 'womens_fashion',
                'product_name': 'Biba Women Salwar Suit',
                'price': '₹1,799',
                'discount': '64% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_fashion_women_4',
                'text': 'Tokyo Talkies Western Dress - Price: ₹699 (MRP: ₹1,999) 65% off. Midi dress, fit and flare, casual. Rating: 4.2/5',
                'category': 'womens_fashion',
                'product_name': 'Tokyo Talkies Western Dress',
                'price': '₹699',
                'discount': '65% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_fashion_women_5',
                'text': 'Zara Women Jeans - Price: ₹1,499 (MRP: ₹2,999) 50% off. High-waist jeans, skinny fit, dark blue. Rating: 4.3/5',
                'category': 'womens_fashion',
                'product_name': 'Zara Women Jeans',
                'price': '₹1,499',
                'discount': '50% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Tablets
            {
                'id': 'fb_tablet_1',
                'text': 'Samsung Galaxy Tab A8 - Price: ₹14,999 (MRP: ₹19,999) 25% off. 10.5" display, 4GB RAM, 64GB storage. Rating: 4.3/5',
                'category': 'tablet',
                'product_name': 'Samsung Galaxy Tab A8',
                'price': '₹14,999',
                'discount': '25% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_tablet_2',
                'text': 'Lenovo Tab M10 Plus - Price: ₹12,999 (MRP: ₹16,999) 24% off. 10.6" FHD, 4GB RAM, quad speakers. Rating: 4.2/5',
                'category': 'tablet',
                'product_name': 'Lenovo Tab M10 Plus',
                'price': '₹12,999',
                'discount': '24% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_tablet_3',
                'text': 'Apple iPad 9th Gen - Price: ₹29,990 (MRP: ₹32,900) 9% off. 10.2" Retina display, A13 chip, iPadOS. Rating: 4.7/5',
                'category': 'tablet',
                'product_name': 'Apple iPad 9th Gen',
                'price': '₹29,990',
                'discount': '9% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Cameras
            {
                'id': 'fb_camera_1',
                'text': 'Canon EOS 1500D DSLR - Price: ₹31,990 (MRP: ₹41,995) 24% off. 24.1MP, WiFi, 18-55mm lens. Rating: 4.5/5',
                'category': 'camera',
                'product_name': 'Canon EOS 1500D DSLR',
                'price': '₹31,990',
                'discount': '24% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_camera_2',
                'text': 'Nikon D3500 DSLR - Price: ₹33,950 (MRP: ₹45,950) 26% off. 24.2MP, guide mode, 18-55mm lens. Rating: 4.6/5',
                'category': 'camera',
                'product_name': 'Nikon D3500 DSLR',
                'price': '₹33,950',
                'discount': '26% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_camera_3',
                'text': 'GoPro Hero 11 Black - Price: ₹39,990 (MRP: ₹54,990) 27% off. Action camera, 5.3K video, waterproof. Rating: 4.7/5',
                'category': 'camera',
                'product_name': 'GoPro Hero 11 Black',
                'price': '₹39,990',
                'discount': '27% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Kitchen Appliances
            {
                'id': 'fb_kitchen_1',
                'text': 'Prestige Induction Cooktop - Price: ₹1,999 (MRP: ₹3,995) 50% off. 1200W, preset menu, auto shut-off. Rating: 4.3/5',
                'category': 'kitchen',
                'product_name': 'Prestige Induction Cooktop',
                'price': '₹1,999',
                'discount': '50% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_kitchen_2',
                'text': 'Philips Mixer Grinder - Price: ₹3,499 (MRP: ₹6,995) 50% off. 750W, 3 jars, turbo function. Rating: 4.4/5',
                'category': 'kitchen',
                'product_name': 'Philips Mixer Grinder',
                'price': '₹3,499',
                'discount': '50% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_kitchen_3',
                'text': 'Kent RO Water Purifier - Price: ₹12,999 (MRP: ₹18,000) 28% off. 8L storage, RO+UV+UF, TDS controller. Rating: 4.5/5',
                'category': 'kitchen',
                'product_name': 'Kent RO Water Purifier',
                'price': '₹12,999',
                'discount': '28% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_kitchen_4',
                'text': 'Wonderchef Electric Kettle - Price: ₹899 (MRP: ₹1,995) 55% off. 1.7L, stainless steel, auto shut-off. Rating: 4.2/5',
                'category': 'kitchen',
                'product_name': 'Wonderchef Electric Kettle',
                'price': '₹899',
                'discount': '55% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Toys & Games
            {
                'id': 'fb_toys_1',
                'text': 'Hot Wheels Track Set - Price: ₹1,299 (MRP: ₹2,499) 48% off. Racing track, 2 cars included, stunts. Rating: 4.4/5',
                'category': 'toys',
                'product_name': 'Hot Wheels Track Set',
                'price': '₹1,299',
                'discount': '48% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_toys_2',
                'text': 'Lego City Police Station - Price: ₹3,999 (MRP: ₹5,999) 33% off. 743 pieces, 6+ years, minifigures. Rating: 4.6/5',
                'category': 'toys',
                'product_name': 'Lego City Police Station',
                'price': '₹3,999',
                'discount': '33% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_toys_3',
                'text': 'Monopoly Board Game - Price: ₹799 (MRP: ₹1,299) 38% off. Classic edition, family game, 8+ years. Rating: 4.5/5',
                'category': 'toys',
                'product_name': 'Monopoly Board Game',
                'price': '₹799',
                'discount': '38% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Books
            {
                'id': 'fb_books_1',
                'text': 'Atomic Habits by James Clear - Price: ₹399 (MRP: ₹599) 33% off. Self-help, bestseller, paperback. Rating: 4.8/5',
                'category': 'books',
                'product_name': 'Atomic Habits',
                'price': '₹399',
                'discount': '33% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_books_2',
                'text': 'The Psychology of Money - Price: ₹299 (MRP: ₹450) 34% off. Finance, Morgan Housel, bestseller. Rating: 4.7/5',
                'category': 'books',
                'product_name': 'The Psychology of Money',
                'price': '₹299',
                'discount': '34% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_books_3',
                'text': 'Think Like a Monk - Price: ₹349 (MRP: ₹499) 30% off. Self-help, Jay Shetty, inspirational. Rating: 4.6/5',
                'category': 'books',
                'product_name': 'Think Like a Monk',
                'price': '₹349',
                'discount': '30% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Sports & Fitness
            {
                'id': 'fb_sports_1',
                'text': 'Nivia Storm Football - Price: ₹599 (MRP: ₹999) 40% off. Size 5, rubber moulded, training ball. Rating: 4.2/5',
                'category': 'sports',
                'product_name': 'Nivia Storm Football',
                'price': '₹599',
                'discount': '40% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_sports_2',
                'text': 'Cosco Yoga Mat - Price: ₹499 (MRP: ₹999) 50% off. 6mm thick, anti-slip, with carry bag. Rating: 4.3/5',
                'category': 'sports',
                'product_name': 'Cosco Yoga Mat',
                'price': '₹499',
                'discount': '50% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_sports_3',
                'text': 'Strauss Adjustable Dumbbells - Price: ₹1,299 (MRP: ₹2,499) 48% off. 10kg pair, chrome finish, home gym. Rating: 4.4/5',
                'category': 'sports',
                'product_name': 'Strauss Adjustable Dumbbells',
                'price': '₹1,299',
                'discount': '48% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_sports_4',
                'text': 'Yonex Badminton Racket - Price: ₹899 (MRP: ₹1,799) 50% off. Graphite frame, lightweight, with cover. Rating: 4.5/5',
                'category': 'sports',
                'product_name': 'Yonex Badminton Racket',
                'price': '₹899',
                'discount': '50% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Beauty & Personal Care
            {
                'id': 'fb_beauty_1',
                'text': 'Lakme Eyeconic Kajal - Price: ₹149 (MRP: ₹225) 34% off. Smudge-proof, long-lasting, deep black. Rating: 4.4/5',
                'category': 'beauty',
                'product_name': 'Lakme Eyeconic Kajal',
                'price': '₹149',
                'discount': '34% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_beauty_2',
                'text': 'Maybelline Fit Me Foundation - Price: ₹399 (MRP: ₹599) 33% off. Natural finish, SPF 18, multiple shades. Rating: 4.5/5',
                'category': 'beauty',
                'product_name': 'Maybelline Fit Me Foundation',
                'price': '₹399',
                'discount': '33% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_beauty_3',
                'text': 'Philips Trimmer BT3221 - Price: ₹1,399 (MRP: ₹2,495) 44% off. Cordless, 20 length settings, 60min runtime. Rating: 4.3/5',
                'category': 'beauty',
                'product_name': 'Philips Trimmer BT3221',
                'price': '₹1,399',
                'discount': '44% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_beauty_4',
                'text': 'Dove Hair Fall Rescue Shampoo - Price: ₹299 (MRP: ₹425) 30% off. 650ml, reduces hair fall, nutrilock. Rating: 4.4/5',
                'category': 'beauty',
                'product_name': 'Dove Hair Fall Rescue Shampoo',
                'price': '₹299',
                'discount': '30% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            
            # Bags & Luggage
            {
                'id': 'fb_bags_1',
                'text': 'Skybags Backpack 30L - Price: ₹999 (MRP: ₹2,299) 57% off. Laptop compartment, water-resistant, multiple pockets. Rating: 4.3/5',
                'category': 'bags',
                'product_name': 'Skybags Backpack 30L',
                'price': '₹999',
                'discount': '57% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_bags_2',
                'text': 'American Tourister Trolley Bag - Price: ₹3,499 (MRP: ₹6,800) 49% off. 55cm cabin size, 4 wheels, hard case. Rating: 4.5/5',
                'category': 'bags',
                'product_name': 'American Tourister Trolley',
                'price': '₹3,499',
                'discount': '49% off',
                'source': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'fb_bags_3',
                'text': 'Wildcraft Duffle Bag 55L - Price: ₹1,799 (MRP: ₹3,495) 49% off. Travel bag, water-resistant, adjustable strap. Rating: 4.4/5',
                'category': 'bags',
                'product_name': 'Wildcraft Duffle Bag',
                'price': '₹1,799',
                'discount': '49% off',
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
            
            # Handle zero vectors
            if np.sum(np.abs(embedding)) < 1e-10:
                embedding = np.random.randn(self.vector_dimension) * 0.01
            
            metadata = {
                'text': doc['text'][:1000],  # Limit text length
                'category': doc['category'],
                'source': doc.get('source', 'unknown'),
                'timestamp': doc.get('timestamp', datetime.now().isoformat())
            }
            
            if 'product_name' in doc:
                metadata['product_name'] = doc['product_name'][:500]
            if 'price' in doc:
                metadata['price'] = str(doc['price'])[:100]
            if 'discount' in doc:
                metadata['discount'] = str(doc['discount'])[:100]
            if 'product_url' in doc:
                metadata['product_url'] = doc['product_url'][:500]
            
            vectors.append({
                'id': doc['id'],
                'values': embedding.tolist(),
                'metadata': metadata
            })
        
        # Batch upsert
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
        
        # Try keyword search first
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
            
            # Format matches
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
            
            # Prefer keyword search if Pinecone scores are low
            if formatted_matches and formatted_matches[0]['score'] < 0.3:
                if keyword_results and keyword_results[0]['score'] > 0:
                    print("  [Debug] Keyword search has better results")
                    return keyword_results
            
            return formatted_matches[:top_k] if formatted_matches else keyword_results
            
        except Exception as e:
            print(f"  [Debug] Pinecone query error: {e}")
            return keyword_results
    
    def _keyword_search(self, query: str, top_k: int = 5, max_price: Optional[int] = None):
        """Keyword-based fallback search"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
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
            
            # Calculate overlap
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
                    if len(word) > 2:
                        if word in product_name_lower:
                            overlap += 4
            
            if overlap > 0:
                score_val = overlap / max(len(query_words), 1)
                scores.append({
                    'id': doc['id'],
                    'score': score_val,
                    'metadata': doc
                })
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
    
    def generate_response(self, query: str, retrieved_docs: list) -> str:
        """Generate response"""
        if not retrieved_docs:
            return ("No specific products found. Try different keywords or ask about:\n"
                   "• 'smartphones under 15000'\n"
                   "• 'best laptops for students'\n"
                   "• 'delivery policy'\n")
        
        # Check for valid results
        has_valid_results = any(doc.get('score', 0) > 0.01 for doc in retrieved_docs)
        
        if not has_valid_results:
            return ("No specific products found. Try different keywords or ask about:\n"
                   "• 'smartphones under 15000'\n"
                   "• 'best laptops for students'\n"
                   "• 'delivery policy'\n")
        
        response = []
        
        for i, doc in enumerate(retrieved_docs[:5], 1):
            text = doc['metadata'].get('text', '')
            score = doc.get('score', 0)
            
            if not text or score <= 0.01:
                continue
            
            response.append(f"{i}. {text}")
            
            # Add product URL if available
            product_url = doc['metadata'].get('product_url', '')
            if product_url and product_url.startswith('http'):
                response.append(f"   🔗 {product_url}")
            
            response.append("")  # Empty line
        
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
