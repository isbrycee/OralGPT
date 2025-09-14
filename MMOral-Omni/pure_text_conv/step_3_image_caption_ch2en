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

class ImageCaptionTranslator:
    def __init__(self):
        # API configuration
        self.api_key = os.getenv("API_KEY", "sk-demo123456789abcdef0123456789abcdef0123456789abcdef")
        self.api_host = os.getenv("API_HOST", "api.example.com")
        self.model = os.getenv("API_MODEL", "gpt-4-turbo")
        
        # Path configuration
        self.base_dir = "/data/workspace/documents/processed"
        
        # Build translation prompt directly in code
        self.translation_prompt = self.build_translation_prompt()
        
        # Get all folders
        self.folders = self.get_folders()
        
    def build_translation_prompt(self) -> str:
        """Build translation prompt"""
        prompt = """You are a document image caption translator and descriptor. The input is text extracted via OCR from the caption or title of a figure/table in a document.

Your tasks:
1. First, determine if the text contains substantive content or if it's just promotional/tutorial content.
2. If it contains meaningful knowledge, translate and describe what the image shows.
3. Remove figure/table numbers and provide a descriptive translation.

IMPORTANT GUIDELINES:

Return "null" ONLY for these types of non-content:
- QR code scanning instructions
- App download instructions
- Publisher/copyright information
- General tutorial steps unrelated to main content
- Contact information or website URLs
- Book purchasing or access instructions

DO NOT return "null" for meaningful content, even if it has figure/table numbers:
- Figure captions with substantive content
- Descriptive content
- Procedural descriptions
- Comparative findings
- Technical images
- Device descriptions

For valid content:
1. Remove the figure/table number
2. Translate the descriptive content
3. Make it a clear description of what the image/table shows
4. Preserve all technical terminology accurately
5. Use descriptive language like "This image shows...", "Cross-section of...", "Comparison between...", etc.

Examples:

Input: "Step 2: Scan the QR code"
Output: null

Input: "Download APP for more resources"
Output: null

Input: "Figure 1-1 Initial process demonstration"
Output: This image shows the initial process demonstration.

Input: "Figure 1-2 Cross-sectional view"
Output: Cross-sectional view of the structure.

Input: "Table 1-1 Classification table"
Output: Classification table of categories.

Input: "Figure 2-3 Before and after comparison"
Output: Comparison of before and after procedure.

Remember: Only return "null" for promotional/tutorial content. For meaningful figures/tables, always provide a descriptive English translation without the numbering."""

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
        """Call API for translation and description"""
        try:
            payload = json.dumps({
                "model": self.model,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": self.translation_prompt
                    },
                    {
                        "role": "user", 
                        "content": f"Caption text: {caption}"
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000
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
            
            # Extract translation result
            if 'choices' in response_data and len(response_data['choices']) > 0:
                choice = response_data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    result = choice['message']['content'].strip()
                    
                    # Check if judged as promotional/tutorial content
                    if result.lower() in ['null', 'none', '']:
                        logger.info(f"      Model judged as promotional/tutorial content: {caption[:50]}...")
                        return None
                    
                    logger.info(f"      Translation description: {result[:80]}...")
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
            output_path = os.path.join(folder_path, "image_translated.json")
            
            # Check if content_list file found
            if not content_list_path:
                logger.warning(f"No content_list file found in folder {folder_name}")
                return False
            
            logger.info(f"Found file: {os.path.basename(content_list_path)}")
            
            # Check if output file already exists
            if os.path.exists(output_path):
                logger.info(f"Translation result for folder {folder_name} already exists, skipping")
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
                logger.info(f"No image captions to translate in folder {folder_name}")
                # Create empty translation file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                return True
            
            logger.info(f"Found {len(image_items)} images to translate in folder {folder_name}")
            
            # Translate each image caption
            translated_results = []
            total_captions = 0
            translated_captions = 0
            skipped_captions = 0
            
            for i, item in enumerate(image_items, 1):
                try:
                    img_path = item.get("img_path", "")
                    captions = item.get("image_caption", [])
                    
                    logger.info(f"  Translating image {i}/{len(image_items)}: {img_path}")
                    
                    valid_translated_captions = []
                    for j, caption in enumerate(captions):
                        if caption.strip():
                            total_captions += 1
                            logger.info(f"    Processing caption {j+1}/{len(captions)}: {caption[:50]}...")
                            
                            try:
                                translated_description = self.call_api(caption)
                                if translated_description:  # If model returns valid description
                                    valid_translated_captions.append(translated_description)
                                    translated_captions += 1
                                else:  # If model judges as promotional/tutorial content
                                    skipped_captions += 1
                                    
                            except Exception as e:
                                logger.error(f"    Translation failed: {e}")
                                # Skip this caption on failure
                                continue
                            
                            # API call interval
                            time.sleep(1)
                    
                    # Only save images with valid translations
                    if valid_translated_captions:
                        result_item = {
                            "image_path": img_path,
                            "image_caption": valid_translated_captions
                        }
                        translated_results.append(result_item)
                        logger.info(f"  Image {img_path} completed, kept {len(valid_translated_captions)} valid descriptions")
                    else:
                        logger.info(f"  Image {img_path} has no valid content-related descriptions")
                
                except Exception as e:
                    logger.error(f"  Error processing image {i}: {e}")
                    continue
            
            # Save translation results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(translated_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Folder {folder_name} processing completed:")
            logger.info(f"  - Total captions: {total_captions}")
            logger.info(f"  - Translation successful: {translated_captions}")
            logger.info(f"  - Skipped (promotional/tutorial): {skipped_captions}")
            logger.info(f"  - Saved images: {len(translated_results)}")
            
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
        
        logger.info(f"All translation results saved in image_translated.json files in each folder")

def main():
    """Main function"""
    try:
        translator = ImageCaptionTranslator()
        translator.process_all_folders()
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
