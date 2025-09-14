import os
import json
import time
import http.client
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageCaptionRewriter:
    def __init__(self):
        # API configuration
        self.api_key = os.getenv("API_KEY", "sk-demo123456789abcdef0123456789abcdef0123456789abcdef")
        self.api_host = os.getenv("API_HOST", "api.example.com")
        self.model = os.getenv("API_MODEL", "gpt-4-turbo")
        
        # Path configuration
        self.base_dir = "/data/workspace/documents/processed"
        
        # Build rewrite prompt directly in code
        self.rewrite_prompt = self.build_rewrite_prompt()
        
        # Get all folders
        self.folders = self.get_folders()
        
    def build_rewrite_prompt(self) -> str:
        """Build rewrite prompt"""
        prompt = """You are a document image caption rewriter. Your task is to transform existing English image captions into natural, descriptive sentences that focus purely on the content, removing any figure/table numbers or references.

IMPORTANT GUIDELINES:
1. REMOVE all figure/table numbers and references (e.g., "Figure 4", "Fig 1-1", "Table 2", etc.)
2. Transform the remaining content into a complete, natural sentence describing the image
3. Vary your sentence structure and wording to create diversity
4. Keep all technical terminology accurate and unchanged
5. Make descriptions sound natural and conversational while maintaining professionalism
6. Don't use repetitive sentence patterns - be creative with structure
7. Focus only on what the image actually shows, not its numbering or position in a document
8. Create complete sentences that flow naturally

Examples of proper transformations:

Input: "Figure 4: Cross-sectional view of structural component"
Output: The cross-sectional view reveals the complex structure of the component.

Input: "Fig 1-1 Comparison of process before and after procedure"
Output: Here we can see a comparison showing the process before and after the procedure.

Input: "Table 2: Classification table of common categories"
Output: A comprehensive table categorizing the most common categories found in the system.

Input: "Figure 3-2: Initial stage in process formation"
Output: The image captures the initial stage where the process begins to form.

Input: "Fig. 5 Diagram showing component positioning"
Output: This diagram clearly displays the positioning of the components.

Input: "Figure A: Detailed section of structure"
Output: A detailed section provides insight into the internal structure.

Input: "Table 1-3: Component with various features"
Output: The component exhibits extensive features affecting multiple aspects.

Input: "Fig 2.1: Assembly technique demonstration"
Output: The step-by-step technique demonstrates proper assembly procedures.

Input: "Figure 6: Comparison between standard and modified versions"
Output: The comparison illustrates the distinct differences between standard and modified versions.

Input: "Table 4-1: Different types of components"
Output: Various types of components are presented showing their unique characteristics.

Remember: 
- Always remove figure/table numbers completely
- Create natural, flowing descriptions that vary in structure 
- Maintain technical accuracy
- Focus on the actual content, not the reference numbering
- Don't use the same sentence pattern repeatedly"""

        return prompt
    
    def find_content_list_file(self, folder_path: str) -> Optional[str]:
        """Find content_list JSON file in folder"""
        auto_path = os.path.join(folder_path, "auto")
        if not os.path.exists(auto_path):
            return None
        
        try:
            for filename in os.listdir(auto_path):
                if filename.endswith('_content_list.json'):
                    return os.path.join(auto_path, filename)
            
            # Also check for direct content_list.json file
            direct_path = os.path.join(auto_path, "content_list.json")
            if os.path.exists(direct_path):
                return direct_path
                
            return None
        except Exception as e:
            logger.error(f"Error finding content_list file: {e}")
            return None
    
    def get_folders(self) -> List[str]:
        """Get all folders to process"""
        folders = []
        try:
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path):
                    # Check if has auto subfolder and content_list file
                    content_list_file = self.find_content_list_file(item_path)
                    if content_list_file:
                        folders.append(item)
                        logger.debug(f"Found folder: {item}, content_list file: {os.path.basename(content_list_file)}")
            
            logger.info(f"Found {len(folders)} folders to process")
            return folders
        except Exception as e:
            logger.error(f"Error getting folder list: {e}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_api(self, caption: str) -> Optional[str]:
        """Call API for rewriting"""
        try:
            payload = json.dumps({
                "model": self.model,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": self.rewrite_prompt
                    },
                    {
                        "role": "user", 
                        "content": f"Please rewrite this caption into a natural description without any figure/table numbers: {caption}"
                    }
                ],
                "temperature": 0.8,  # Higher temperature for creativity and diversity
                "max_tokens": 500
            })
            
            conn = http.client.HTTPSConnection(self.api_host, timeout=30)
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            conn.request("POST", "/v1/chat/completions", payload, headers)
            response = conn.getresponse()
            response_status = response.status
            data = response.read()
            conn.close()
            
            # Check HTTP status code
            if response_status != 200:
                logger.error(f"HTTP error {response_status}")
                raise Exception(f"HTTP error {response_status}")
            
            # Parse JSON response
            response_data = json.loads(data.decode("utf-8"))
            
            # Extract rewrite result
            if 'choices' in response_data and len(response_data['choices']) > 0:
                choice = response_data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    result = choice['message']['content'].strip()
                    
                    logger.info(f"      Rewrite result: {result[:80]}...")
                    return result
            
            logger.error(f"API response format error: {response_data}")
            raise Exception("API response format error")
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def process_content_list(self, folder_name: str) -> bool:
        """Process content_list.json file for single folder"""
        try:
            # Build file paths
            folder_path = os.path.join(self.base_dir, folder_name)
            content_list_path = self.find_content_list_file(folder_path)
            output_path = os.path.join(folder_path, "image_rewritten.json")
            
            # Check if content_list file found
            if not content_list_path:
                logger.warning(f"No content_list file found in folder {folder_name}")
                return False
            
            logger.info(f"Found file: {os.path.basename(content_list_path)}")
            
            # Check if output file already exists
            if os.path.exists(output_path):
                logger.info(f"Rewrite result for folder {folder_name} already exists, skipping")
                return True
            
            # Read content_list file
            with open(content_list_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            # Filter image type items with captions
            image_items = []
            for item in content_list:
                if (item.get("type") == "image" and 
                    item.get("image_caption") and 
                    len(item.get("image_caption", [])) > 0):
                    image_items.append(item)
            
            if not image_items:
                logger.info(f"No image captions to rewrite in folder {folder_name}")
                # Create empty rewrite file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                return True
            
            logger.info(f"Found {len(image_items)} images to rewrite in folder {folder_name}")
            
            # Rewrite each image caption
            rewritten_results = []
            total_captions = 0
            rewritten_captions = 0
            
            for i, item in enumerate(image_items, 1):
                try:
                    img_path = item.get("img_path", "")
                    captions = item.get("image_caption", [])
                    
                    logger.info(f"  Rewriting image {i}/{len(image_items)}: {img_path}")
                    
                    rewritten_caption_list = []
                    for j, caption in enumerate(captions):
                        if caption.strip():
                            total_captions += 1
                            logger.info(f"    Processing caption {j+1}/{len(captions)}: {caption[:50]}...")
                            
                            try:
                                rewritten_description = self.call_api(caption)
                                if rewritten_description:
                                    rewritten_caption_list.append(rewritten_description)
                                    rewritten_captions += 1
                                else:
                                    # If API call fails, keep original caption
                                    rewritten_caption_list.append(caption)
                                    rewritten_captions += 1
                                    logger.info(f"    Using original caption: {caption}")
                                    
                            except Exception as e:
                                logger.error(f"    Rewrite failed: {e}")
                                # Keep original caption on failure
                                rewritten_caption_list.append(caption)
                                rewritten_captions += 1
                                logger.info(f"    Using original caption: {caption}")
                            
                            # API call interval
                            time.sleep(1)
                    
                    # Save rewritten image information
                    if rewritten_caption_list:
                        result_item = {
                            "image_path": img_path,
                            "image_caption": rewritten_caption_list
                        }
                        rewritten_results.append(result_item)
                        logger.info(f"  Image {img_path} completed, rewritten {len(rewritten_caption_list)} descriptions")
                
                except Exception as e:
                    logger.error(f"  Error processing image {i}: {e}")
                    continue
            
            # Save rewrite results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(rewritten_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Folder {folder_name} processing completed:")
            logger.info(f"  - Total captions: {total_captions}")
            logger.info(f"  - Rewrite successful: {rewritten_captions}")
            logger.info(f"  - Saved images: {len(rewritten_results)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing folder {folder_name}: {e}")
            return False
    
    def process_all_folders(self) -> None:
        """Process all folders"""
        logger.info(f"Starting to process {len(self.folders)} folders")
        
        success_count = 0
        failed_folders = []
        
        for i, folder_name in enumerate(self.folders, 1):
            logger.info(f"Processing folder {i}/{len(self.folders)}: {folder_name}")
            
            try:
                if self.process_content_list(folder_name):
                    success_count += 1
                    logger.info(f"Folder {folder_name} processed successfully")
                else:
                    failed_folders.append(folder_name)
                    logger.warning(f"Folder {folder_name} processing failed")
                
            except Exception as e:
                failed_folders.append(folder_name)
                logger.error(f"Folder {folder_name} processing exception: {e}")
            
            # Interval between folders
            if i < len(self.folders):
                logger.info(f"Waiting 2 seconds before processing next folder...")
                time.sleep(2)
        
        # Show final statistics
        logger.info("=" * 60)
        logger.info(f"Processing completed! Success: {success_count}/{len(self.folders)}")
        if failed_folders:
            logger.warning(f"Failed folders: {failed_folders}")
        
        logger.info(f"All rewrite results saved in image_rewritten.json files in each folder")

def main():
    """Main function"""
    try:
        rewriter = ImageCaptionRewriter()
        rewriter.process_all_folders()
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
