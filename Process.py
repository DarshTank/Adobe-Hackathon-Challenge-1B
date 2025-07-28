import os
import json
import fitz  # PyMuPDF
import re
import unicodedata
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import math
from statistics import mean, median, mode, StatisticsError
import datetime
import sys

# --- New Imports for Challenge 1b ---
# These libraries are required for semantic analysis.
# You can install them using: pip install sentence-transformers torch
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
except ImportError:
    print("Dependencies for Challenge 1b not found.", file=sys.stderr)
    print("Please run: pip install sentence-transformers torch", file=sys.stderr)
    sys.exit(1)
# --- End of New Imports ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedPDFExtractor:
    """
    This class is an enhanced version of the solution for Challenge 1a.
    It extracts text, analyzes structure, and identifies potential headings.
    A new method 'get_heading_snippets_single_pass' has been added to support Challenge 1b.
    """
    def __init__(self):
        self.min_heading_candidates = 3
        self.max_analysis_pages = 15
        self.title_area_threshold = 0.4
        self.max_title_length = 200

    def extract_pdf_metadata(self, pdf_path):
        """Enhanced metadata extraction with better span processing"""
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF file not found: {pdf_path}")
            return []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open or read PDF {pdf_path}: {e}")
            return []

        all_spans = []
        for page_number, page in enumerate(doc, start=1):
            try:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block.get('type') == 0:  # text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "")
                                if not text.strip():
                                    continue

                                font = span.get("font", "")
                                size = span.get("size", 0)
                                bbox = span.get("bbox", [0, 0, 0, 0])
                                flags = span.get("flags", 0)

                                is_bold = "Bold" in font or bool(flags & 16)
                                is_italic = "Italic" in font or bool(flags & 2)

                                all_spans.append({
                                    "page": page_number,
                                    "text": text.strip(),
                                    "font": font,
                                    "size": size,
                                    "bbox": bbox,
                                    "bold": is_bold,
                                    "italic": is_italic,
                                    "flags": flags
                                })
            except Exception as e:
                logger.warning(f"Could not process page {page_number} in {pdf_path}: {e}")
                continue
        
        doc.close()
        return all_spans

    def get_heading_snippets_single_pass(self, pdf_path):
        """
        Extracts potential headings and their following text snippets in a single pass.
        This is optimized for Challenge 1b to provide context for semantic ranking.
        """
        spans = self.extract_pdf_metadata(pdf_path)
        if not spans:
            return [], "No Title Found"

        structure_data = self.analyze_document_structure(spans)
        learned_patterns = self.learn_heading_patterns(structure_data)
        title = self.extract_title_enhanced(spans, learned_patterns)
        headings = self.extract_headings_enhanced(spans, learned_patterns, title)

        # Create a quick lookup for headings by page and text
        heading_map = defaultdict(list)
        for h in headings:
            heading_map[h['page']].append(h)

        snippets = []
        for i, heading in enumerate(headings):
            page_num = heading['page']
            
            # Find the start and end y-coordinates for the snippet
            start_y = heading['bbox'][3] # Bottom of the current heading
            end_y = float('inf')

            # Find the next heading on the same page to define the snippet boundary
            next_heading_on_page = None
            current_heading_y = heading['bbox'][1]
            
            sorted_headings_on_page = sorted(heading_map[page_num], key=lambda x: x['bbox'][1])

            for j, h_on_page in enumerate(sorted_headings_on_page):
                if h_on_page['text'] == heading['text'] and abs(h_on_page['bbox'][1] - current_heading_y) < 5:
                   if j + 1 < len(sorted_headings_on_page):
                       next_heading_on_page = sorted_headings_on_page[j+1]
                       break
            
            if next_heading_on_page:
                end_y = next_heading_on_page['bbox'][1] # Top of the next heading

            # Extract text within the snippet boundaries
            page_spans = [s for s in spans if s['page'] == page_num and start_y <= s['bbox'][1] < end_y]
            page_spans.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
            
            snippet_text = ' '.join([s['text'] for s in page_spans])
            
            # Clean and limit snippet length
            snippet_text = re.sub(r'\s+', ' ', snippet_text).strip()
            if len(snippet_text) > 1000: # Limit context size
                snippet_text = snippet_text[:1000] + "..."

            snippets.append({
                "document": os.path.basename(pdf_path),
                "page": page_num,
                "section_title": heading['text'],
                "content_snippet": f"{heading['text']}. {snippet_text}" # Prepend title for context
            })
            
        return snippets, title

    def analyze_document_structure(self, spans):
        """Enhanced document structure analysis"""
        structure_data = {
            'font_analysis': defaultdict(list),
            'size_distribution': defaultdict(int),
            'position_analysis': defaultdict(list),
            'text_patterns': [],
            'page_analysis': defaultdict(list),
            'line_spacing': [],
            'formatting_patterns': defaultdict(int)
        }
        
        # Analyze all spans
        for span in spans:
            font_key = (span["font"], span["size"], span["flags"])
            structure_data['font_analysis'][font_key].append(span)
            structure_data['size_distribution'][span["size"]] += 1
            structure_data['page_analysis'][span["page"]].append(span)
            
            # Extract text patterns
            patterns = self.extract_text_patterns(span["text"])
            structure_data['text_patterns'].extend(patterns)
            
            # Formatting analysis
            if span["bold"]:
                structure_data['formatting_patterns']['bold'] += 1
            if span["italic"]:
                structure_data['formatting_patterns']['italic'] += 1
            if span["text"].isupper():
                structure_data['formatting_patterns']['uppercase'] += 1
        
        # Calculate size statistics
        all_sizes = list(structure_data['size_distribution'].keys())
        if all_sizes:
            structure_data['size_stats'] = {
                'min': min(all_sizes),
                'max': max(all_sizes),
                'median': median(all_sizes),
                'mean': mean(all_sizes),
                'unique_sizes': len(all_sizes)
            }
        
        return structure_data

    def extract_text_patterns(self, text):
        """Enhanced pattern extraction with more comprehensive rules"""
        patterns = []
        
        if not text or not text.strip():
            return patterns
        
        text_clean = text.strip()
        
        # Numbering patterns
        numbering_patterns = [
            (r'^\d+\.?\s*', 'decimal_number'),
            (r'^\d+\.\d+\.?\s*', 'decimal_subsection'),
            (r'^\d+\.\d+\.\d+\.?\s*', 'decimal_subsubsection'),
            (r'^\(\d+\)\s*', 'parenthetical_number'),
            (r'^\[\d+\]\s*', 'bracketed_number'),
            (r'^[IVXLCDMivxlcdm]+\.?\s*', 'roman_numeral'),
            (r'^[A-Za-z]\.?\s*', 'letter_enumeration'),
            (r'^\([A-Za-z]\)\s*', 'parenthetical_letter')
        ]
        
        for pattern, pattern_type in numbering_patterns:
            match = re.match(pattern, text_clean)
            if match:
                patterns.append((pattern_type, match.group()))
        
        # Bullet patterns
        bullet_chars = ['•', '·', '▪', '▫', '◦', '‣', '⁃', '-', '–', '—', '○', '●', '□', '■']
        if text_clean and text_clean[0] in bullet_chars:
            patterns.append(('bullet_point', text_clean[0]))
        
        # Formatting patterns
        if text_clean.isupper() and len(text_clean.split()) <= 10:
            patterns.append(('all_uppercase', 'CAPS'))
        
        if text_clean.endswith(':') and len(text_clean) < 100:
            patterns.append(('colon_ending', 'COLON'))
        
        # Special heading indicators
        heading_indicators = ['chapter', 'section', 'part', 'appendix', 'introduction', 'conclusion']
        text_lower = text_clean.lower()
        for indicator in heading_indicators:
            if text_lower.startswith(indicator):
                patterns.append(('heading_keyword', indicator))
        
        return patterns

    def learn_heading_patterns(self, structure_data):
        """Enhanced pattern learning with better font analysis"""
        font_analysis = structure_data['font_analysis']
        size_stats = structure_data.get('size_stats', {})
        
        # Analyze each font combination
        font_characteristics = {}
        
        for font_key, spans in font_analysis.items():
            font, size, flags = font_key
            
            if not spans:
                continue
                
            texts = [span['text'] for span in spans]
            avg_length = mean([len(text) for text in texts]) if texts else 0
            avg_words = mean([len(text.split()) for text in texts]) if texts else 0
            unique_ratio = len(set(texts)) / len(texts) if texts else 0
            
            # Position analysis
            positions = [(span['bbox'][0], span['bbox'][1]) for span in spans]
            pages = [span['page'] for span in spans]
            
            font_characteristics[font_key] = {
                'count': len(spans),
                'avg_length': avg_length,
                'avg_words': avg_words,
                'unique_ratio': unique_ratio,
                'size': size,
                'is_bold': bool(flags & 16),
                'is_italic': bool(flags & 2),
                'texts': texts,
                'positions': positions,
                'pages': pages,
                'size_percentile': self.calculate_size_percentile(size, structure_data['size_distribution'])
            }
        
        # Score fonts for heading likelihood
        heading_candidates = []
        
        for font_key, char in font_characteristics.items():
            score = self.calculate_heading_score(char, size_stats)
            
            if score >= 5:  # Minimum threshold for heading consideration
                heading_candidates.append((font_key, char, score))
        
        # Sort by score
        heading_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Pattern analysis
        pattern_stats = defaultdict(int)
        for pattern_type, pattern_value in structure_data['text_patterns']:
            pattern_stats[pattern_type] += 1
        
        return {
            'heading_fonts': heading_candidates,
            'pattern_frequency': pattern_stats,
            'size_stats': size_stats,
            'total_fonts': len(font_characteristics)
        }

    def calculate_size_percentile(self, size, size_distribution):
        """Calculate what percentile this font size is"""
        if not size_distribution: return 0
        total_occurrences = sum(size_distribution.values())
        if total_occurrences == 0: return 0
        smaller_sizes = sum(count for s, count in size_distribution.items() if s < size)
        return (smaller_sizes / total_occurrences) * 100

    def calculate_heading_score(self, characteristics, size_stats):
        """Enhanced scoring algorithm for heading detection"""
        score = 0
        
        # Size scoring (more important factor)
        size_percentile = characteristics['size_percentile']
        if size_percentile >= 80:
            score += 5
        elif size_percentile >= 60:
            score += 3
        elif size_percentile >= 40:
            score += 1
        
        # Style scoring
        if characteristics['is_bold']:
            score += 3
        if characteristics['is_italic']:
            score += 1
        
        # Text characteristics
        if characteristics['avg_words'] <= 8:  # Short text
            score += 2
        elif characteristics['avg_words'] <= 15:
            score += 1
        elif characteristics['avg_words'] > 25:
            score -= 2
        
        # Uniqueness (headings are usually unique)
        if characteristics['unique_ratio'] > 0.8:
            score += 3
        elif characteristics['unique_ratio'] > 0.5:
            score += 1
        elif characteristics['unique_ratio'] < 0.2:
            score -= 2
        
        # Frequency analysis (headings appear less frequently)
        if characteristics['count'] <= 5:
            score += 2
        elif characteristics['count'] <= 15:
            score += 1
        elif characteristics['count'] >= 50:
            score -= 3
        
        # Length analysis
        if characteristics['avg_length'] <= 50:
            score += 1
        elif characteristics['avg_length'] >= 150:
            score -= 2
        
        return score

    def extract_title_enhanced(self, spans, learned_patterns):
        """Enhanced title extraction combining both approaches"""
        # Filter to first page spans
        page1_spans = [s for s in spans if s["page"] == 1 and len(s["text"].strip()) > 2]
        
        if not page1_spans:
            return "Untitled Document"
        
        # Define title area (top portion of first page)
        max_y = max(s["bbox"][3] for s in page1_spans) if page1_spans else 0
        title_threshold = max_y * self.title_area_threshold
        title_area_spans = [s for s in page1_spans if s["bbox"][1] < title_threshold]
        
        if not title_area_spans:
            title_area_spans = page1_spans[:10]  # Fallback to first 10 spans
        
        # Get title candidates using multiple methods
        title_candidates = []
        
        # Method 1: Largest font sizes
        size_groups = defaultdict(list)
        for span in title_area_spans:
            size_key = round(span["size"] * 2) / 2
            size_groups[size_key].append(span)
        
        # Take top 3 font sizes
        sorted_sizes = sorted(size_groups.keys(), reverse=True)[:3]
        
        for size in sorted_sizes:
            spans_for_size = size_groups[size]
            lines = self.group_spans_into_lines(spans_for_size)
            
            for line_spans in lines:
                # Filter out labels and colons
                content_spans = [s for s in line_spans if not s["text"].strip().endswith(":")]
                if content_spans:
                    merged_text = self.merge_line_spans(content_spans)
                    if merged_text and len(merged_text.strip()) >= 5:
                        score = self.score_title_candidate(merged_text, size, spans_for_size[0])
                        title_candidates.append((merged_text.strip(), score, size))
        
        # Method 2: Use learned patterns for validation
        if learned_patterns and 'heading_fonts' in learned_patterns:
            for font_key, char, font_score in learned_patterns['heading_fonts'][:3]:
                for span in title_area_spans:
                    span_key = (span["font"], span["size"], span["flags"])
                    if span_key == font_key:
                        text = span["text"].strip()
                        if len(text) >= 5:
                            score = self.score_title_candidate(text, span["size"], span)
                            title_candidates.append((text, score + font_score/2, span["size"]))
        
        if not title_candidates:
            return "Untitled Document"
        
        # Sort by score and select best candidate
        title_candidates.sort(key=lambda x: x[1], reverse=True)
        best_title = title_candidates[0][0]
        
        # Clean and validate title
        cleaned_title = self.clean_title(best_title)
        
        # Check length constraint
        if len(cleaned_title) > self.max_title_length:
            # Try to get a shorter version
            words = cleaned_title.split()
            if len(words) > 3:
                cleaned_title = ' '.join(words[:int(len(words)*0.7)])
        
        return cleaned_title if cleaned_title else "Untitled Document"

    def score_title_candidate(self, text, font_size, span):
        """Score a title candidate"""
        score = 0
        
        # Length scoring
        word_count = len(text.split())
        if 3 <= word_count <= 12:
            score += 5
        elif word_count <= 20:
            score += 2
        elif word_count > 25:
            score -= 5
        
        # Font size (assume larger is better for title)
        score += font_size / 5
        
        # Style scoring
        if span.get("bold", False):
            score += 3
        if span.get("italic", False):
            score += 1
        
        # Position scoring (higher on page is better)
        y_pos = span["bbox"][1]
        if y_pos < 100:  # Very top of page
            score += 5
        elif y_pos < 200:
            score += 3
        
        # Content quality
        if not any(exclude in text.lower() for exclude in ['page', 'figure', 'table', 'www', 'http', '©']):
            score += 2
        
        # Capitalization patterns
        if text.istitle():
            score += 2
        elif text.isupper() and len(text) < 100:
            score += 1
        
        return score

    def extract_headings_enhanced(self, spans, learned_patterns, title_text=""):
        """Enhanced heading extraction using learned patterns and positioning"""
        if not spans:
            return []
        
        # Get heading font candidates from learned patterns
        heading_fonts = {}
        if learned_patterns and 'heading_fonts' in learned_patterns:
            for font_key, char, score in learned_patterns['heading_fonts']:
                heading_fonts[font_key] = score
        
        # Extract title font sizes to exclude from headings
        title_font_sizes = self.get_title_font_sizes(spans, title_text)
        
        heading_candidates = []
        
        # Process each page
        max_page = max(s["page"] for s in spans) if spans else 0
        for page_num in range(1, max_page + 1):
            page_spans = [s for s in spans if s["page"] == page_num]
            if not page_spans:
                continue
            
            # Sort by vertical position
            page_spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
            
            i = 0
            while i < len(page_spans):
                span = page_spans[i]
                text = span["text"].strip()
                
                if len(text) < 2:
                    i += 1
                    continue
                
                # Check if this could be a heading
                font_key = (span["font"], span["size"], span["flags"])
                is_heading, confidence = self.is_potential_heading_enhanced(
                    span, page_spans, heading_fonts, title_font_sizes, learned_patterns
                )
                
                if is_heading:
                    # Try to build complete heading (handle multi-line)
                    complete_heading = self.build_complete_heading(page_spans, i)
                    
                    if complete_heading:
                        heading_candidates.append({
                            "text": complete_heading["text"],
                            "size": span["size"],
                            "page": page_num,
                            "confidence": confidence,
                            "bbox": span["bbox"],
                            "font_key": font_key,
                            "bold": span["bold"],
                            "spans_used": complete_heading["spans_count"]
                        })
                        
                        # Skip the spans we used
                        i += complete_heading["spans_count"]
                    else:
                        i += 1
                else:
                    i += 1
        
        if not heading_candidates:
            return []
        
        # Remove duplicates and assign levels
        unique_headings = self.remove_duplicate_headings(heading_candidates)
        final_headings = self.assign_heading_levels_enhanced(unique_headings)
        
        return final_headings

    def is_potential_heading_enhanced(self, span, page_spans, heading_fonts, title_font_sizes, learned_patterns):
        """Enhanced heading detection with multiple criteria"""
        text = span["text"].strip()
        font_key = (span["font"], span["size"], span["flags"])
        
        # Basic validation
        if len(text) < 2 or len(text) > 300:
            return False, 0
        
        confidence = 0
        
        # Check against learned heading fonts
        if font_key in heading_fonts:
            confidence += heading_fonts[font_key]
        
        # Size analysis
        size = span["size"]
        if any(abs(size - title_size) <= 1.0 for title_size in title_font_sizes):
            return False, 0  # Too similar to title
        
        size_percentile = self.calculate_span_size_percentile(span, page_spans)
        if size_percentile >= 70:
            confidence += 4
        elif size_percentile >= 50:
            confidence += 2
        
        # Style analysis
        if span["bold"]:
            confidence += 3
        if span["italic"]:
            confidence += 1
        
        # Text pattern analysis
        patterns = self.extract_text_patterns(text)
        for pattern_type, pattern_value in patterns:
            if pattern_type in ['decimal_number', 'roman_numeral', 'heading_keyword']:
                confidence += 2
            elif pattern_type in ['all_uppercase', 'colon_ending']:
                confidence += 1
        
        # Length analysis
        word_count = len(text.split())
        if word_count <= 10:
            confidence += 2
        elif word_count <= 20:
            confidence += 1
        elif word_count > 30:
            confidence -= 3
        
        # Position and context analysis
        following_content = self.analyze_following_content(span, page_spans)
        if following_content["has_substantial_content"]:
            confidence += 3
        if following_content["content_smaller_font"]:
            confidence += 2
        
        # Content validation
        if self.is_likely_non_heading_content(text):
            confidence -= 5
        
        return confidence >= 6, confidence

    def calculate_span_size_percentile(self, span, page_spans):
        """Calculate size percentile for a span within page context"""
        sizes = [s["size"] for s in page_spans]
        if not sizes:
            return 50
        
        smaller_count = sum(1 for s in sizes if s < span["size"])
        return (smaller_count / len(sizes)) * 100

    def analyze_following_content(self, heading_span, page_spans):
        """Analyze content following a potential heading"""
        heading_y = heading_span["bbox"][3]  # Bottom of heading
        heading_size = heading_span["size"]
        
        following_spans = []
        for span in page_spans:
            if span["bbox"][1] > heading_y + (heading_size * 0.5):  # Below heading with some gap
                following_spans.append(span)
        
        # Sort by position
        following_spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        
        # Analyze first few spans
        analysis = {
            "has_substantial_content": False,
            "content_smaller_font": False,
            "total_chars": 0,
            "unique_texts": 0
        }
        
        if len(following_spans) >= 3:
            first_spans = following_spans[:8]
            texts = [s["text"].strip() for s in first_spans if len(s["text"].strip()) > 2]
            
            analysis["total_chars"] = sum(len(t) for t in texts)
            analysis["unique_texts"] = len(set(texts))
            analysis["has_substantial_content"] = analysis["total_chars"] >= 50
            
            # Check if following content has smaller font
            following_sizes = [s["size"] for s in first_spans if s["text"].strip()]
            if following_sizes:
                avg_following_size = mean(following_sizes)
                analysis["content_smaller_font"] = avg_following_size < heading_size
        
        return analysis

    def is_likely_non_heading_content(self, text):
        """Check if text is likely NOT a heading"""
        text_lower = text.lower()
        
        # Skip patterns
        skip_patterns = [
            'copyright', '©', 'page', 'figure', 'table', 'www', 'http', '.com',
            'email', '@', 'phone', 'tel:', 'fax:', 'address', 'signature',
            'name:', 'date:', 'time:', 'location:'
        ]
        
        if any(pattern in text_lower for pattern in skip_patterns):
            return True
        
        # Skip if mostly numbers or symbols
        if re.match(r'^[\d\s\-_\.,:;()]+$', text):
            return True
        
        # Skip very repetitive text
        words = text.split()
        if len(words) > 1 and len(set(words)) == 1:
            return True
        
        return False

    def build_complete_heading(self, page_spans, start_index):
        """Build complete heading that might span multiple lines"""
        if start_index >= len(page_spans):
            return None
        
        start_span = page_spans[start_index]
        heading_text = start_span["text"].strip()
        spans_used = 1
        
        # Check for continuation on same line
        current_y = start_span["bbox"][1]
        current_size = start_span["size"]
        
        i = start_index + 1
        while i < len(page_spans):
            next_span = page_spans[i]
            y_diff = abs(next_span["bbox"][1] - current_y)
            size_diff = abs(next_span["size"] - current_size)
            
            # Same line continuation
            if y_diff < current_size * 0.4 and size_diff < 1:
                heading_text = self.merge_with_overlap_removal(heading_text, next_span["text"].strip())
                spans_used += 1
                i += 1
            else:
                break
        
        # Check for multi-line continuation
        if i < len(page_spans):
            next_line_span = page_spans[i]
            y_gap = next_line_span["bbox"][1] - current_y
            
            # Next line with same size might be continuation
            if (current_size * 0.8 < y_gap < current_size * 2.5 and 
                abs(next_line_span["size"] - current_size) < 1 and
                len(heading_text.split()) < 8):  # Only for shorter headings
                
                continuation_text = next_line_span["text"].strip()
                if continuation_text and not continuation_text.startswith(('•', '-', '1.', 'a.', 'i.')):
                    heading_text += " " + continuation_text
                    spans_used += 1
        
        return {
            "text": heading_text,
            "spans_count": spans_used
        }

    def remove_duplicate_headings(self, candidates):
        """Remove duplicate headings while preserving the best ones"""
        seen_texts = {}
        unique_candidates = []
        
        # Sort by confidence first
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        for candidate in candidates:
            text_key = candidate["text"].lower().strip()
            
            # Check for exact match
            if text_key in seen_texts:
                continue
            
            # Check for substantial overlap
            is_duplicate = False
            for existing_text in seen_texts:
                similarity = self.calculate_text_similarity(text_key, existing_text)
                if similarity > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts[text_key] = candidate
                unique_candidates.append(candidate)
        
        return unique_candidates

    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def assign_heading_levels_enhanced(self, candidates):
        """Enhanced level assignment with better hierarchy detection"""
        if not candidates:
            return []
        
        # Sort by size (descending) then by page order
        candidates.sort(key=lambda x: (-x["size"], x["page"], x["bbox"][1]))
        
        # Get unique sizes
        unique_sizes = sorted(list(set(c["size"] for c in candidates)), reverse=True)
        
        # Assign levels (maximum 6 levels)
        size_to_level = {}
        for i, size in enumerate(unique_sizes[:6]):
            size_to_level[size] = f"H{i+1}"
        
        # Build final outline
        outline = []
        for candidate in candidates:
            level = size_to_level.get(candidate["size"], "H6") # Default to H6
            
            # Re-sort by page and position for final output
            final_candidate_list = sorted(candidates, key=lambda x: (x["page"], x["bbox"][1]))

        for candidate in final_candidate_list:
            level = size_to_level.get(candidate['size'], 'H6')
            outline.append({
                "level": level,
                "text": candidate["text"],
                "page": candidate["page"],
                "confidence": candidate.get("confidence", 0),
                "bbox": candidate.get("bbox", [])
            })
        return outline


    def get_title_font_sizes(self, spans, title_text=""):
        """Get font sizes used in title area"""
        page1_spans = [s for s in spans if s["page"] == 1]
        if not page1_spans:
            return []
        
        max_y = max(s["bbox"][3] for s in page1_spans) if page1_spans else 0
        title_threshold = max_y * self.title_area_threshold
        title_spans = [s for s in page1_spans if s["bbox"][1] < title_threshold and s["size"] > 8]
        
        title_sizes = []
        if title_text:
            # Find spans that match title text
            for span in title_spans:
                if title_text.lower() in span["text"].lower() or span["text"].lower() in title_text.lower():
                    title_sizes.append(span["size"])
        
        if not title_sizes:
            # Fallback: get largest sizes from title area
            if title_spans:
                sizes = [s["size"] for s in title_spans]
                title_sizes = sorted(set(sizes), reverse=True)[:2]
        
        return title_sizes

    def group_spans_into_lines(self, spans):
        """Group spans into lines based on vertical position"""
        if not spans:
            return []
        
        spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        
        lines = []
        current_line = [spans[0]]
        
        for span in spans[1:]:
            prev_y = current_line[-1]["bbox"][1]
            curr_y = span["bbox"][1]
            y_diff = abs(curr_y - prev_y)
            
            font_size = span["size"]
            line_threshold = font_size * 0.4
            
            if y_diff < line_threshold:
                current_line.append(span)
            else:
                lines.append(current_line)
                current_line = [span]
        
        if current_line:
            lines.append(current_line)
        
        return lines

    def merge_line_spans(self, spans):
        """Merge spans on same line with overlap removal"""
        if not spans:
            return ""
        
        if len(spans) == 1:
            return spans[0]["text"]
        
        spans.sort(key=lambda s: s["bbox"][0])  # Sort by x position
        
        merged_text = spans[0]["text"].strip()
        
        for span in spans[1:]:
            next_text = span["text"].strip()
            merged_text = self.merge_with_overlap_removal(merged_text, next_text)
        
        return merged_text

    def merge_with_overlap_removal(self, existing_text, new_text):
        """Merge text while removing overlaps"""
        if not existing_text:
            return new_text
        if not new_text:
            return existing_text
        
        # Check complete containment
        if new_text.lower() in existing_text.lower():
            return existing_text
        if existing_text.lower() in new_text.lower():
            return new_text
        
        # Find character-level overlap
        max_overlap_len = min(len(existing_text), len(new_text))
        best_overlap_len = 0
        
        for i in range(1, max_overlap_len + 1):
            end_part = existing_text[-i:].lower()
            start_part = new_text[:i].lower()
            
            if end_part == start_part:
                best_overlap_len = i
        
        if best_overlap_len > 0:
            return existing_text + new_text[best_overlap_len:]
        
        # Check word-level overlap
        existing_words = existing_text.split()
        new_words = new_text.split()
        
        if existing_words and new_words:
            if existing_words[-1].lower() == new_words[0].lower():
                return existing_text + " " + " ".join(new_words[1:])
        
        # No overlap found
        if existing_text and new_text:
            if existing_text[-1].isalnum() and new_text[0].isalnum():
                return existing_text + " " + new_text
            else:
                return existing_text + new_text
        
        return existing_text + new_text

    def clean_title(self, title):
        """Clean and validate extracted title"""
        if not title:
            return "Untitled Document"
        
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove leading/trailing artifacts
        title = re.sub(r'^[\s\-_]+|[\s\-_]+$', '', title)
        title = title.replace('\n', ' ').replace('\r', ' ')
        
        # Remove excessive repeated characters
        title = re.sub(r'(.)\1{4,}', r'\1', title)
        
        # Remove common artifacts
        title = re.sub(r'\s*\|\s*', ' ', title)  # Remove pipe separators
        title = re.sub(r'\s*[•·▪]\s*', ' ', title)  # Remove bullets
        
        if len(title.strip()) < 3:
            return "Untitled Document"
        
        return title.strip()

class SemanticRanker:
    """
    Handles the semantic analysis for Challenge 1b.
    It loads a sentence transformer model, processes document snippets,
    and ranks them based on relevance to a given query.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the SemanticRanker.
        Loads the sentence-transformer model. Using a smaller, efficient model
        is key to meeting the performance and resource constraints.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}'.")
            logger.error("Please ensure you have an internet connection for the first download,")
            logger.error("or that the model is available in your local cache.")
            logger.error(f"Error: {e}")
            sys.exit(1)

    def rank_sections(self, query, sections, top_k=10):
        """
        Ranks sections based on semantic similarity to the query.

        Args:
            query (str): The search query (combination of persona and job).
            sections (list of dict): A list of sections, each with a 'content_snippet'.
            top_k (int): The number of top results to return.

        Returns:
            list of dict: The ranked list of sections, with an added 'score'.
        """
        if not sections:
            return []

        # Encode the query and all section snippets
        corpus = [s['content_snippet'] for s in sections]
        query_embedding = self.model.encode(query, convert_to_tensor=True, device=self.device)
        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True, device=self.device)

        # Compute cosine similarity
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        # Add scores to the original sections
        for i, section in enumerate(sections):
            section['score'] = cos_scores[i].item()

        # Sort by score in descending order
        ranked_sections = sorted(sections, key=lambda x: x['score'], reverse=True)

        return ranked_sections[:top_k]

def run_challenge_1b(input_json_path, output_json_path):
    """
    Main function to execute Challenge 1b logic.
    """
    logger.info(f"--- Starting Challenge 1b for {os.path.basename(input_json_path)} ---")

    # 1. Load Input Data
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {input_json_path}. File may be empty or malformed. Details: {e}")
        return
    except Exception as e:
        logger.error(f"Could not read input JSON {input_json_path}: {e}")
        return

    # More robust check for input data
    if not isinstance(input_data, dict) or not input_data:
        logger.error(f"Invalid or empty JSON object in {input_json_path}.")
        return

    # --- UPDATED PARSING LOGIC ---
    persona_data = input_data.get("persona")
    persona = persona_data.get("role") if isinstance(persona_data, dict) else persona_data

    job_data = input_data.get("job") or input_data.get("job to be done")
    job = job_data.get("task") if isinstance(job_data, dict) else job_data

    documents_data = input_data.get("documents", [])
    documents = [doc.get("filename") for doc in documents_data if isinstance(doc, dict) and "filename" in doc]
    # --- END OF UPDATED PARSING LOGIC ---
    
    # Check for missing or empty required fields
    if not all([persona, job, documents]):
        logger.error(f"Input JSON {input_json_path} is missing required fields or they are empty.")
        logger.error(f"  - Found persona: {persona}")
        logger.error(f"  - Found job: {job}")
        logger.error(f"  - Found documents: {documents}")
        return

    query = f"{persona}: {job}"
    logger.info(f"Constructed query: {query}")
    
    # Get the directory of the input JSON to resolve relative PDF paths
    base_dir = os.path.dirname(input_json_path)

    # 2. Initialize Tools
    extractor = EnhancedPDFExtractor()
    ranker = SemanticRanker()

    # 3. Extract Snippets from all Documents
    all_snippets = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # --- CORRECTED PATH LOGIC ---
        future_to_pdf = {
            executor.submit(extractor.get_heading_snippets_single_pass, os.path.join(base_dir, 'PDFs', doc)): doc 
            for doc in documents
        }
        # --- END OF CORRECTED PATH LOGIC ---
        for future in as_completed(future_to_pdf):
            pdf_name = future_to_pdf[future]
            try:
                snippets, _ = future.result()
                if snippets:
                    all_snippets.extend(snippets)
                else:
                    logger.warning(f"No snippets extracted from {pdf_name}")
            except Exception as e:
                logger.error(f"Failed to extract snippets from {pdf_name}: {e}")

    if not all_snippets:
        logger.error("No snippets could be extracted from any of the provided documents. Aborting.")
        return

    logger.info(f"Extracted a total of {len(all_snippets)} snippets from {len(documents)} documents.")

    # 4. Rank Snippets
    ranked_sections = ranker.rank_sections(query, all_snippets, top_k=10)
    logger.info(f"Ranked {len(ranked_sections)} sections based on relevance.")

    # 5. Format Output
    output_data = {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job": job,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": [],
        "sub_section_analysis": [] # This part is left for potential future enhancement
    }

    for i, section in enumerate(ranked_sections):
        output_data["extracted_sections"].append({
            "document": section["document"],
            "page_number": section["page"],
            "section_title": section["section_title"],
            "importance_rank": i + 1
        })

    # 6. Save Output
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Successfully generated output at: {output_json_path}")
    except Exception as e:
        logger.error(f"Failed to write output JSON to {output_json_path}: {e}")

    logger.info(f"--- Finished Challenge 1b for {os.path.basename(input_json_path)} ---")


if __name__ == "__main__":
    # This allows running the script for a specific collection from the command line.
    # Example: python mainc2.py path/to/your/Collection_1
    if len(sys.argv) > 1:
        # User has provided a path
        collection_path = sys.argv[1]
        logger.info(f"Running on specified collection: {collection_path}")
        if os.path.isdir(collection_path):
            input_json = os.path.join(collection_path, 'challenge1b_input.json')
            output_json = os.path.join(collection_path, 'challenge1b_output_generated.json')
            if os.path.exists(input_json):
                run_challenge_1b(input_json, output_json)
            else:
                logger.error(f"Input file not found: {input_json}")
        else:
            logger.error(f"Provided path is not a directory: {collection_path}")
    else:
        # Default behavior: run on all collections found in the current directory
        base_challenge_dir = '.' # Assume collections are in the current directory
        logger.info(f"No specific collection path provided. Searching for Collection directories in '{base_challenge_dir}'...")
        if os.path.isdir(base_challenge_dir):
            for item in sorted(os.listdir(base_challenge_dir)):
                collection_path = os.path.join(base_challenge_dir, item)
                if os.path.isdir(collection_path) and item.startswith("Collection"):
                    input_json = os.path.join(collection_path, 'challenge1b_input.json')
                    output_json = os.path.join(collection_path, 'challenge1b_output_generated.json')
                    if os.path.exists(input_json):
                        run_challenge_1b(input_json, output_json)
        else:
            logger.warning(f"Default directory '{base_challenge_dir}' not found.")
