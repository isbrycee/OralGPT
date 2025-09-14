#!/usr/bin/env python3
import os
import re
import json
import time
import http.client
from typing import List, Dict, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import math
import pickle
import hashlib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranslationError(Exception):
    """Custom translation error class"""
    def __init__(self, message: str, error_type: str = "unknown", details: Dict = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.now()

class DocumentTranslator:
    def __init__(self):
        # API configuration
        self.api_key = os.getenv("API_KEY", "sk-demo123456789abcdef0123456789abcdef0123456789abcdef")
        self.api_host = os.getenv("API_HOST", "api.example.com")
        self.model = os.getenv("API_MODEL", "gpt-4-turbo")
        
        # Path configuration
        self.input_dir = "/data/workspace/documents/input"
        self.output_dir = "/data/workspace/documents/output"
        self.prompt_file = "/data/workspace/documents/translation_prompt.txt"
        
        # Checkpoint configuration
        self.checkpoint_dir = "/data/workspace/documents/checkpoints"
        self.completed_files_file = os.path.join(self.checkpoint_dir, "completed_files.json")
        self.section_cache_dir = os.path.join(self.checkpoint_dir, "sections")
        self.error_log_file = os.path.join(self.checkpoint_dir, "translation_errors.json")
        self.error_report_file = os.path.join(self.checkpoint_dir, "error_report.txt")
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.section_cache_dir, exist_ok=True)
        
        # Load translation prompt
        self.translation_prompt = self.load_translation_prompt()
        
        # Retry configuration
        self.max_section_retries = 3
        self.retry_delay = 5
        
        # Load completed files list
        self.completed_files = self.load_completed_files()
        
        # Error tracking
        self.translation_errors = self.load_error_log()
        
    def load_error_log(self) -> List[Dict]:
        """Load error log"""
        try:
            if os.path.exists(self.error_log_file):
                with open(self.error_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Error loading error log: {e}")
            return []
    
    def save_error_log(self):
        """Save error log"""
        try:
            with open(self.error_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.translation_errors, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving error log: {e}")
    
    def record_error(self, error: TranslationError, context: Dict = None):
        """Record translation error"""
        error_record = {
            'timestamp': error.timestamp.isoformat(),
            'error_type': error.error_type,
            'message': str(error),
            'details': error.details,
            'context': context or {}
        }
        
        self.translation_errors.append(error_record)
        self.save_error_log()
        
        logger.error(f"Recorded error: {error.error_type} - {str(error)}")
    
    def generate_error_report(self):
        """Generate error report"""
        if not self.translation_errors:
            return
        
        try:
            # Group by error type
            error_by_type = {}
            for error in self.translation_errors:
                error_type = error['error_type']
                if error_type not in error_by_type:
                    error_by_type[error_type] = []
                error_by_type[error_type].append(error)
            
            # Generate report
            report_lines = [
                "=" * 80,
                f"Translation Error Report - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 80,
                f"Total errors: {len(self.translation_errors)}",
                f"Error types: {len(error_by_type)}",
                ""
            ]
            
            # List details by error type
            for error_type, errors in error_by_type.items():
                report_lines.extend([
                    f"Error type: {error_type}",
                    f"Count: {len(errors)}",
                    "-" * 50
                ])
                
                for i, error in enumerate(errors[:10], 1):
                    report_lines.extend([
                        f"{i}. Time: {error['timestamp']}",
                        f"   Message: {error['message']}",
                        f"   Details: {error['details']}",
                        f"   Context: {error.get('context', {})}",
                        ""
                    ])
                
                if len(errors) > 10:
                    report_lines.append(f"   ... {len(errors) - 10} more similar errors")
                
                report_lines.append("")
            
            # Write report file
            with open(self.error_report_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"Error report generated: {self.error_report_file}")
            
        except Exception as e:
            logger.error(f"Error generating error report: {e}")
    
    def load_translation_prompt(self) -> str:
        """Load translation prompt from file"""
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            error = TranslationError(f"Prompt file not found: {self.prompt_file}", "file_not_found")
            self.record_error(error)
            raise
        except Exception as e:
            error = TranslationError(f"Error reading prompt file: {e}", "file_read_error", {"file": self.prompt_file})
            self.record_error(error)
            raise
    
    def load_completed_files(self) -> set:
        """Load completed files list"""
        try:
            if os.path.exists(self.completed_files_file):
                with open(self.completed_files_file, 'r', encoding='utf-8') as f:
                    completed_list = json.load(f)
                    logger.info(f"Loaded completed files list, {len(completed_list)} files")
                    return set(completed_list)
            else:
                logger.info("No completed files list found, starting from beginning")
                return set()
        except Exception as e:
            error = TranslationError(f"Error loading completed files list: {e}", "checkpoint_load_error")
            self.record_error(error)
            return set()
    
    def save_completed_file(self, filename: str):
        """Save completed file"""
        self.completed_files.add(filename)
        try:
            with open(self.completed_files_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.completed_files), f, ensure_ascii=False, indent=2)
            logger.info(f"Saved completed file record: {filename}")
        except Exception as e:
            error = TranslationError(f"Error saving completed file record: {e}", "checkpoint_save_error", {"filename": filename})
            self.record_error(error)
    
    def get_section_hash(self, title: str, content: str) -> str:
        """Generate hash for section content"""
        section_text = f"{title}||{content}"
        return hashlib.md5(section_text.encode('utf-8')).hexdigest()
    
    def get_section_cache_path(self, book_title: str, section_hash: str) -> str:
        """Get section cache file path"""
        return os.path.join(self.section_cache_dir, f"{book_title}_{section_hash}.pkl")
    
    def save_section_translation(self, book_title: str, title: str, content: str, translation: str):
        """Save section translation to cache"""
        section_hash = self.get_section_hash(title, content)
        cache_path = self.get_section_cache_path(book_title, section_hash)
        
        cache_data = {
            'title': title,
            'content': content,
            'translation': translation,
            'timestamp': time.time()
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cached section translation: {title[:30]}...")
        except Exception as e:
            error = TranslationError(f"Error saving section cache: {e}", "cache_save_error", 
                                   {"book_title": book_title, "section_title": title[:50]})
            self.record_error(error)
    
    def load_section_translation(self, book_title: str, title: str, content: str) -> Optional[str]:
        """Load section translation from cache"""
        section_hash = self.get_section_hash(title, content)
        cache_path = self.get_section_cache_path(book_title, section_hash)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if (cache_data['title'] == title and 
                cache_data['content'] == content and 
                cache_data['translation']):
                logger.info(f"Loaded section translation from cache: {title[:30]}...")
                return cache_data['translation']
            else:
                os.remove(cache_path)
                return None
                
        except Exception as e:
            error = TranslationError(f"Error loading section cache: {e}", "cache_load_error",
                                   {"book_title": book_title, "section_title": title[:50]})
            self.record_error(error)
            try:
                os.remove(cache_path)
            except:
                pass
            return None
    
    def clean_section_cache(self, book_title: str):
        """Clean section cache for specified book"""
        try:
            for filename in os.listdir(self.section_cache_dir):
                if filename.startswith(f"{book_title}_") and filename.endswith('.pkl'):
                    cache_path = os.path.join(self.section_cache_dir, filename)
                    os.remove(cache_path)
            logger.info(f"Cleaned section cache for book {book_title}")
        except Exception as e:
            error = TranslationError(f"Error cleaning section cache: {e}", "cache_cleanup_error", {"book_title": book_title})
            self.record_error(error)
    
    def is_file_completed(self, filename: str) -> bool:
        """Check if file is completed"""
        book_title = os.path.splitext(filename)[0]
        
        if filename in self.completed_files:
            output_file = os.path.join(self.output_dir, f"{book_title}.txt")
            if os.path.exists(output_file):
                return True
            else:
                self.completed_files.discard(filename)
                self.save_completed_files_list()
                return False
        
        return False
    
    def save_completed_files_list(self):
        """Save completed files list"""
        try:
            with open(self.completed_files_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.completed_files), f, ensure_ascii=False, indent=2)
        except Exception as e:
            error = TranslationError(f"Error saving completed files list: {e}", "checkpoint_save_error")
            self.record_error(error)
    
    def parse_api_error(self, response_data: Dict) -> Tuple[str, str, Dict]:
        """Parse API error response"""
        error_type = "api_response_error"
        error_message = "Unknown API error"
        error_details = {}
        
        if 'error' in response_data:
            error_info = response_data['error']
            if isinstance(error_info, dict):
                error_type = error_info.get('type', 'api_error')
                error_message = error_info.get('message', 'No error message')
                error_details = {
                    'code': error_info.get('code'),
                    'param': error_info.get('param'),
                    'type': error_info.get('type')
                }
            else:
                error_message = str(error_info)
        elif 'message' in response_data:
            error_message = response_data['message']
        else:
            error_message = f"API response format error: {response_data}"
            error_details = {"response": response_data}
        
        return error_type, error_message, error_details
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_api(self, content: str, book_title: str, content_type: str = "content") -> str:
        """Call API for translation"""
        context = {
            "book_title": book_title,
            "content_type": content_type,
            "content_length": len(content),
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }
        
        try:
            if content_type == "title":
                user_message = f"【Document Title】: {book_title}\n\nThis is a section title, please translate:\n{content}"
            else:
                user_message = f"【Document Title】: {book_title}\n\n{content}"
            
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
                        "content": user_message
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 4000
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
            response_reason = response.reason
            data = response.read()
            conn.close()
            
            if response_status != 200:
                error_type = f"http_{response_status}"
                error_message = f"HTTP error {response_status}: {response_reason}"
                error_details = {
                    "status_code": response_status,
                    "reason": response_reason,
                    "response_body": data.decode("utf-8", errors="ignore")[:500]
                }
                
                error = TranslationError(error_message, error_type, error_details)
                self.record_error(error, context)
                raise error
            
            try:
                response_data = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as e:
                error_message = f"JSON decode error: {e}"
                error_details = {
                    "raw_response": data.decode("utf-8", errors="ignore")[:500],
                    "json_error": str(e)
                }
                
                error = TranslationError(error_message, "json_decode_error", error_details)
                self.record_error(error, context)
                raise error
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                choice = response_data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    translated_text = choice['message']['content']
                    if translated_text and translated_text.strip():
                        return translated_text.strip()
                    else:
                        error_message = "API returned empty translation"
                        error_details = {"response": response_data}
                        
                        error = TranslationError(error_message, "empty_translation", error_details)
                        self.record_error(error, context)
                        raise error
                else:
                    error_message = "Missing message.content field in API response"
                    error_details = {"response": response_data}
                    
                    error = TranslationError(error_message, "missing_content_field", error_details)
                    self.record_error(error, context)
                    raise error
            else:
                error_type, error_message, error_details = self.parse_api_error(response_data)
                
                error = TranslationError(error_message, error_type, error_details)
                self.record_error(error, context)
                raise error
                
        except TranslationError:
            raise
        except Exception as e:
            error_message = f"API call exception: {str(e)}"
            error_type = "api_call_exception"
            error_details = {
                "exception_type": type(e).__name__,
                "exception_message": str(e)
            }
            
            error = TranslationError(error_message, error_type, error_details)
            self.record_error(error, context)
            raise error
    
    def split_markdown_by_chapters(self, content: str) -> List[Tuple[str, str]]:
        """Split markdown content by chapters"""
        chapter_pattern = r'^(#{1,6})\s*(Chapter\s+\d+.*?)$'
        
        lines = content.split('\n')
        sections = []
        current_section = {'title': '', 'content': []}
        
        for line in lines:
            match = re.match(chapter_pattern, line, re.MULTILINE)
            if match:
                if current_section['title'] or current_section['content']:
                    section_content = '\n'.join(current_section['content']).strip()
                    if section_content:
                        sections.append((current_section['title'], section_content))
                
                current_section = {
                    'title': match.group(2).strip(),
                    'content': []
                }
            else:
                current_section['content'].append(line)
        
        if current_section['title'] or current_section['content']:
            section_content = '\n'.join(current_section['content']).strip()
            if section_content:
                sections.append((current_section['title'], section_content))
        
        if not sections and content.strip():
            sections.append(('Full Text', content.strip()))
        
        logger.info(f"Split into {len(sections)} sections")
        return sections
    
    def translate_section_with_retry(self, title: str, section_content: str, book_title: str, section_index: int, total_sections: int) -> Tuple[Optional[str], Optional[TranslationError]]:
        """Translate section with retry mechanism and cache"""
        cached_translation = self.load_section_translation(book_title, title, section_content)
        if cached_translation:
            logger.info(f"Section {section_index}/{total_sections} loaded from cache: {title[:50]}...")
            return cached_translation, None
        
        last_error = None
        for attempt in range(self.max_section_retries):
            try:
                logger.info(f"Translating section {section_index}/{total_sections}: {title[:50]}... (attempt {attempt + 1}/{self.max_section_retries})")
                
                translated_title = ""
                if title and title != 'Full Text':
                    logger.info(f"  Translating title: {title}")
                    try:
                        translated_title = self.call_api(title, book_title, "title")
                        if not translated_title:
                            translated_title = title
                        time.sleep(1)
                    except TranslationError as e:
                        logger.warning(f"Title translation failed, using original: {e}")
                        translated_title = title
                        last_error = e
                
                if len(section_content) > 3000:
                    sub_sections = self.split_long_content(section_content)
                    translated_sub_sections = []
                    
                    for j, sub_content in enumerate(sub_sections, 1):
                        logger.info(f"    Translating sub-section {j}/{len(sub_sections)}")
                        try:
                            translated_sub = self.call_api(sub_content, book_title, "content")
                            if translated_sub:
                                translated_sub_sections.append(translated_sub)
                            else:
                                raise TranslationError(f"Sub-section {j} translation is empty", "empty_sub_translation")
                            time.sleep(1)
                        except TranslationError as e:
                            last_error = e
                            raise e
                    
                    translated_content = '\n\n'.join(translated_sub_sections)
                else:
                    translated_content = self.call_api(section_content, book_title, "content")
                
                if translated_content:
                    if translated_title:
                        section_text = f"{translated_title}\n\n{translated_content}"
                    else:
                        section_text = translated_content
                    
                    self.save_section_translation(book_title, title, section_content, section_text)
                    
                    logger.info(f"Section {title} translation successful")
                    return section_text, None
                else:
                    raise TranslationError("Translation content is empty", "empty_translation")
                    
            except TranslationError as e:
                last_error = e
                logger.warning(f"Section {title} attempt {attempt + 1} failed: {e.error_type} - {str(e)}")
                if attempt < self.max_section_retries - 1:
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Section {title} failed after {self.max_section_retries} attempts")
                    context = {
                        "book_title": book_title,
                        "section_title": title,
                        "section_index": section_index,
                        "total_attempts": self.max_section_retries
                    }
                    self.record_error(e, context)
                    return None, e
            except Exception as e:
                error = TranslationError(f"Unexpected error: {str(e)}", "unexpected_error", 
                                       {"exception_type": type(e).__name__})
                last_error = error
                logger.error(f"Section {title} unexpected error: {e}")
                if attempt < self.max_section_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    context = {
                        "book_title": book_title,
                        "section_title": title,
                        "section_index": section_index
                    }
                    self.record_error(error, context)
                    return None, error
            
            time.sleep(2)
        
        return None, last_error
    
    def process_single_file(self, file_path: str) -> None:
        """Process single markdown file"""
        filename = os.path.basename(file_path)
        book_title = os.path.splitext(filename)[0]
        
        if self.is_file_completed(filename):
            logger.info(f"File {filename} already completed, skipping")
            return
        
        logger.info(f"Started processing document: {book_title}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"File {book_title} is empty, skipping")
                return
            
            sections = self.split_markdown_by_chapters(content)
            
            translated_sections = []
            failed_sections = []
            
            for i, (title, section_content) in enumerate(sections, 1):
                translated_section, error = self.translate_section_with_retry(title, section_content, book_title, i, len(sections))
                
                if translated_section:
                    translated_sections.append(translated_section)
                else:
                    failed_sections.append((i, title, section_content, error))
            
            if failed_sections:
                logger.info(f"Found {len(failed_sections)} failed sections, starting supplementary translation...")
                
                retry_success_count = 0
                for section_index, title, section_content, original_error in failed_sections:
                    logger.info(f"Supplementary translation for failed section: {title[:50]}... (reason: {original_error.error_type if original_error else 'unknown'})")
                    
                    time.sleep(10)
                    
                    translated_section, retry_error = self.translate_section_with_retry(title, section_content, book_title, section_index, len(sections))
                    
                    if translated_section:
                        translated_sections.insert(section_index - 1, translated_section)
                        retry_success_count += 1
                        logger.info(f"Supplementary translation successful: {title}")
                    else:
                        error_summary = f"{retry_error.error_type}: {str(retry_error)}" if retry_error else "Unknown error"
                        failure_marker = f"【Translation Failed - Original Text】{title}\n\nError: {error_summary}\n\n{section_content}"
                        translated_sections.insert(section_index - 1, failure_marker)
                        logger.error(f"Supplementary translation still failed, keeping original: {title} (error: {error_summary})")
                
                logger.info(f"Supplementary translation completed, successful {retry_success_count}/{len(failed_sections)} sections")
            else:
                retry_success_count = 0
            
            if translated_sections:
                final_content = '\n\n---\n\n'.join(translated_sections)
                
                output_file = os.path.join(self.output_dir, f"{book_title}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(final_content)
                
                self.save_completed_file(filename)
                
                success_rate = (len(sections) - len(failed_sections) + (retry_success_count if failed_sections else 0)) / len(sections) * 100
                logger.info(f"Document {book_title} processing completed, success rate: {success_rate:.1f}%, saved to: {output_file}")
            else:
                error_message = f"Document {book_title} no sections translated successfully"
                error = TranslationError(error_message, "complete_failure", {"total_sections": len(sections)})
                context = {"book_title": book_title, "filename": filename}
                self.record_error(error, context)
                logger.error(error_message)
                
        except Exception as e:
            error_message = f"Error processing file {file_path}: {e}"
            error = TranslationError(error_message, "file_processing_error", 
                                   {"filename": filename, "exception_type": type(e).__name__})
            context = {"file_path": file_path}
            self.record_error(error, context)
            logger.error(error_message)
            self.clean_section_cache(book_title)
    
    def split_long_content(self, content: str, max_length: int = 3000) -> List[str]:
        """Split long content by paragraphs"""
        paragraphs = content.split('\n\n')
        sections = []
        current_section = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            if current_length + paragraph_length > max_length and current_section:
                sections.append('\n\n'.join(current_section))
                current_section = [paragraph]
                current_length = paragraph_length
            else:
                current_section.append(paragraph)
                current_length += paragraph_length + 2
        
        if current_section:
            sections.append('\n\n'.join(current_section))
        
        return sections
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        md_files = [f for f in os.listdir(self.input_dir) if f.endswith('.md')]
        total_files = len(md_files)
        completed_files = len(self.completed_files)
        
        cache_files = [f for f in os.listdir(self.section_cache_dir) if f.endswith('.pkl')]
        cached_sections = len(cache_files)
        
        error_stats = {}
        for error in self.translation_errors:
            error_type = error['error_type']
            error_stats[error_type] = error_stats.get(error_type, 0) + 1
        
        return {
            'total_files': total_files,
            'completed_files': completed_files,
            'remaining_files': total_files - completed_files,
            'cached_sections': cached_sections,
            'completion_rate': completed_files / total_files * 100 if total_files > 0 else 0,
            'total_errors': len(self.translation_errors),
            'error_types': error_stats
        }
    
    def process_all_files(self) -> None:
        """Process all markdown files in single thread"""
        logger.info(f"Started processing directory: {self.input_dir}")
        
        stats = self.get_processing_stats()
        logger.info(f"Processing stats: total_files={stats['total_files']}, completed={stats['completed_files']}, "
                   f"remaining={stats['remaining_files']}, cached_sections={stats['cached_sections']}, "
                   f"completion_rate={stats['completion_rate']:.1f}%, total_errors={stats['total_errors']}")
        
        if stats['error_types']:
            logger.info(f"Error type stats: {stats['error_types']}")
        
        md_files = [f for f in os.listdir(self.input_dir) if f.endswith('.md')]
        
        if not md_files:
            logger.warning(f"No .md files found in directory {self.input_dir}")
            return
        
        remaining_files = [f for f in md_files if not self.is_file_completed(f)]
        
        if not remaining_files:
            logger.info("All files are completed!")
            if self.translation_errors:
                self.generate_error_report()
            return
        
        logger.info(f"Need to process {len(remaining_files)} files (skipped {len(md_files) - len(remaining_files)} completed files)")
        
        for i, filename in enumerate(remaining_files, 1):
            logger.info(f"Processing file {i}/{len(remaining_files)}: {filename}")
            file_path = os.path.join(self.input_dir, filename)
            
            try:
                self.process_single_file(file_path)
                logger.info(f"File {filename} processing completed")
            except Exception as e:
                error_message = f"Error processing file {filename}: {e}"
                error = TranslationError(error_message, "file_processing_error",
                                       {"filename": filename, "exception_type": type(e).__name__})
                context = {"file_index": i, "total_files": len(remaining_files)}
                self.record_error(error, context)
                logger.error(error_message)
                continue
            
            if i < len(remaining_files):
                logger.info(f"Waiting 3 seconds before processing next file...")
                time.sleep(3)
        
        final_stats = self.get_processing_stats()
        logger.info(f"Processing completed! Final stats: total_files={final_stats['total_files']}, "
                   f"completed={final_stats['completed_files']}, completion_rate={final_stats['completion_rate']:.1f}%, "
                   f"total_errors={final_stats['total_errors']}")
        
        if final_stats['error_types']:
            logger.info(f"Final error type stats: {final_stats['error_types']}")
        
        logger.info(f"Results saved in: {self.output_dir}")
        
        if self.translation_errors:
            self.generate_error_report()
            logger.info(f"Error report generated: {self.error_report_file}")

def main():
    """Main function"""
    translator = DocumentTranslator()
    translator.process_all_files()

if __name__ == "__main__":
    main()
