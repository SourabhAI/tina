"""
Preprocessor module for the intent classification system.
Handles text normalization, spelling correction, and ID canonicalization.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import unicodedata

from intent_classifier.utils.helpers import (
    normalize_id,
    normalize_csi_section,
    clean_text,
    SpellingCorrector
)


logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Text preprocessing for intent classification.
    Normalizes text, fixes common typos, and canonicalizes IDs.
    """
    
    def __init__(self, enable_spell_correction: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            enable_spell_correction: Whether to enable spelling correction
        """
        self.enable_spell_correction = enable_spell_correction
        self.spelling_corrector = SpellingCorrector() if enable_spell_correction else None
        
        # Domain-specific abbreviation expansions
        self.abbreviations = {
            'rfi': 'RFI',
            'rfis': 'RFI',
            'cb': 'CB',
            'cbs': 'CB',
            'ahu': 'AHU',
            'vav': 'VAV',
            'mep': 'MEP',
            'hvac': 'HVAC',
            'elec': 'electrical',
            'mech': 'mechanical',
            'struct': 'structural',
            'arch': 'architectural',
            'dwg': 'drawing',
            'dwgs': 'drawings',
            'spec': 'specification',
            'specs': 'specifications',
            'sub': 'submittal',
            'subs': 'submittals',
            'seq': 'sequence',
            'sch': 'schedule',
            'bldg': 'building',
            'rm': 'room',
            'flr': 'floor',
            'lvl': 'level',
            'qty': 'quantity',
            'mat': 'material',
            'mats': 'materials',
            'equip': 'equipment',
            'temp': 'temperature',
            'press': 'pressure',
            'dim': 'dimension',
            'dims': 'dimensions',
            'elev': 'elevation',
            'sect': 'section',
            'dtl': 'detail',
            'dtls': 'details',
            'demo': 'demolition',
            'exist': 'existing',
            'prop': 'proposed',
            'typ': 'typical',
            'alt': 'alternate',
            'addl': 'additional',
            'req': 'required',
            'reqd': 'required',
            'min': 'minimum',
            'max': 'maximum',
            'approx': 'approximately',
            'incl': 'including',
            'excl': 'excluding',
            'w/': 'with',
            'w/o': 'without',
            'thru': 'through',
            'btwn': 'between',
            'adj': 'adjacent',
            'opp': 'opposite',
            'sim': 'similar',
            'ref': 'reference',
            'std': 'standard',
            'mfr': 'manufacturer',
            'mfg': 'manufacturing',
            'inst': 'installation',
            'const': 'construction',
            'coord': 'coordinate',
            'prelim': 'preliminary',
            'rev': 'revision',
            'addm': 'addendum',
            'supp': 'supplement',
            'alt': 'alternate',
            'pnl': 'panel',
            'xfmr': 'transformer',
            'gen': 'generator',
            'sw': 'switch',
            'ckt': 'circuit',
            'bkr': 'breaker',
            'disc': 'disconnect',
            'jb': 'junction box',
            'conc': 'concrete',
            'reinf': 'reinforcement',
            'galv': 'galvanized',
            'alum': 'aluminum',
            'ss': 'stainless steel',
            'cs': 'carbon steel',
            'pvc': 'PVC',
            'cpvc': 'CPVC',
            'hdpe': 'HDPE',
            'ci': 'cast iron',
            'di': 'ductile iron',
            'cu': 'copper',
            'brass': 'brass',
            'chrm': 'chrome',
            'gfrc': 'glass fiber reinforced concrete',
            'gfrp': 'glass fiber reinforced plastic',
            'frp': 'fiber reinforced plastic',
            'gyp': 'gypsum',
            'gwb': 'gypsum wall board',
            'plwd': 'plywood',
            'osb': 'oriented strand board',
            'mdf': 'medium density fiberboard',
            'pt': 'pressure treated',
            'ga': 'gauge',
            'thk': 'thick',
            'rad': 'radius',
            'dia': 'diameter',
            'ctr': 'center',
            'ctrs': 'centers',
            'o.c.': 'on center',
            'oc': 'on center',
            'clr': 'clear',
            'clg': 'ceiling',
            'flg': 'flooring',
            'fdn': 'foundation',
            'ftg': 'footing',
            'col': 'column',
            'bm': 'beam',
            'jst': 'joist',
            'stl': 'steel',
            'conc': 'concrete',
            'mas': 'masonry',
            'cmu': 'concrete masonry unit',
            'brk': 'brick',
            'ins': 'insulation',
            'vapr': 'vapor',
            'barr': 'barrier',
            'memb': 'membrane',
            'sht': 'sheet',
            'pnl': 'panel',
            'bd': 'board',
            'fin': 'finish',
            'pnt': 'paint',
            'ptd': 'painted',
            'galv': 'galvanized',
            'ano': 'anodized',
            'pc': 'piece',
            'pcs': 'pieces',
            'ea': 'each',
            'pr': 'pair',
            'prs': 'pairs',
            'lf': 'linear feet',
            'sf': 'square feet',
            'sy': 'square yards',
            'cf': 'cubic feet',
            'cy': 'cubic yards',
            'gal': 'gallons',
            'lbs': 'pounds',
            'psi': 'pounds per square inch',
            'psf': 'pounds per square foot',
            'plf': 'pounds per linear foot',
            'kips': 'kips',
            'pcf': 'pounds per cubic foot',
            'mph': 'miles per hour',
            'fps': 'feet per second',
            'fpm': 'feet per minute',
            'cfm': 'cubic feet per minute',
            'gpm': 'gallons per minute',
            'btu': 'BTU',
            'btuh': 'BTU per hour',
            'kw': 'kilowatt',
            'hp': 'horsepower',
            'v': 'volt',
            'a': 'amp',
            'hz': 'hertz',
            'ph': 'phase',
            'vac': 'volts AC',
            'vdc': 'volts DC',
            'nec': 'National Electrical Code',
            'ibc': 'International Building Code',
            'imc': 'International Mechanical Code',
            'ipc': 'International Plumbing Code',
            'nfpa': 'NFPA',
            'ul': 'UL',
            'astm': 'ASTM',
            'ansi': 'ANSI',
            'aia': 'AIA',
            'leed': 'LEED',
            'ada': 'ADA',
            'osha': 'OSHA',
            'voc': 'VOC',
            'co2': 'carbon dioxide',
            'co': 'carbon monoxide',
            'no': 'normally open',
            'nc': 'normally closed',
            'hoa': 'hand-off-auto',
            'vfd': 'variable frequency drive',
            'bms': 'building management system',
            'ems': 'energy management system',
            'ddc': 'direct digital control',
            'plc': 'programmable logic controller',
            'rtu': 'rooftop unit',
            'fcu': 'fan coil unit',
            'vav': 'variable air volume',
            'cav': 'constant air volume',
            'erv': 'energy recovery ventilator',
            'hrv': 'heat recovery ventilator',
            'dx': 'direct expansion',
            'chw': 'chilled water',
            'hw': 'hot water',
            'cw': 'condenser water',
            'dcw': 'domestic cold water',
            'dhw': 'domestic hot water',
            'san': 'sanitary',
            'vent': 'ventilation',
            'exh': 'exhaust',
            'oa': 'outside air',
            'ra': 'return air',
            'sa': 'supply air',
            'ea': 'exhaust air',
            'ma': 'mixed air',
            'diff': 'diffuser',
            'grll': 'grille',
            'reg': 'register',
            'dmp': 'damper',
            'fltr': 'filter',
            'coil': 'coil',
            'hx': 'heat exchanger',
            'pump': 'pump',
            'vlv': 'valve',
            'prv': 'pressure relief valve',
            'bfp': 'backflow preventer',
            'exp': 'expansion',
            'comp': 'compressor',
            'cond': 'condenser',
            'evap': 'evaporator',
            'ref': 'refrigerant',
            'ins': 'insulation',
            'jkt': 'jacket',
            'acc': 'access',
            'serv': 'service',
            'maint': 'maintenance',
            'emer': 'emergency',
            'norm': 'normal',
            'stdby': 'standby',
            'aux': 'auxiliary',
            'prim': 'primary',
            'sec': 'secondary',
            'horiz': 'horizontal',
            'vert': 'vertical',
            'long': 'longitudinal',
            'trans': 'transverse',
            'perp': 'perpendicular',
            'parl': 'parallel',
            'adj': 'adjustable',
            'rem': 'removable',
            'perm': 'permanent',
            'temp': 'temporary',
            'fut': 'future',
            'ex': 'existing',
            'demo': 'demolish',
            'reloc': 'relocate',
            'repl': 'replace',
            'prov': 'provide',
            'furn': 'furnish',
            'inst': 'install',
            'conn': 'connect',
            'disc': 'disconnect',
            'rem': 'remove',
            'sal': 'salvage',
            'prot': 'protect',
            'supp': 'support',
            'anch': 'anchor',
            'att': 'attach',
            'sec': 'secure',
            'seal': 'seal',
            'grout': 'grout',
            'flash': 'flashing',
            'caulk': 'caulk',
            'weld': 'weld',
            'bolt': 'bolt',
            'screw': 'screw',
            'nail': 'nail',
            'glue': 'glue',
            'adh': 'adhesive',
            'tape': 'tape',
            'wrap': 'wrap',
            'coat': 'coating',
            'prim': 'primer',
            'sealr': 'sealer',
            'cond': 'condition',
            'qual': 'quality',
            'perf': 'performance',
            'eff': 'efficiency',
            'cap': 'capacity',
            'rat': 'rating',
            'tol': 'tolerance',
            'allow': 'allowable',
            'max': 'maximum',
            'min': 'minimum',
            'avg': 'average',
            'std': 'standard',
            'alt': 'alternate',
            'opt': 'optional',
            'req': 'required',
            'rec': 'recommended',
            'pref': 'preferred',
            'acc': 'acceptable',
            'suit': 'suitable',
            'compat': 'compatible',
            'cert': 'certified',
            'appv': 'approved',
            'list': 'listed',
            'rate': 'rated',
            'test': 'tested',
            'verif': 'verified',
            'insp': 'inspected',
            'rev': 'reviewed',
            'subm': 'submitted',
            'appr': 'approved',
            'rej': 'rejected',
            'pend': 'pending',
            'hold': 'hold',
            'canc': 'cancelled',
            'susp': 'suspended',
            'comp': 'complete',
            'incomp': 'incomplete',
            'part': 'partial',
            'prog': 'progress',
            'sched': 'scheduled',
            'plan': 'planned',
            'est': 'estimated',
            'approx': 'approximate',
            'tent': 'tentative',
            'prelim': 'preliminary',
            'final': 'final',
            'orig': 'original',
            'rev': 'revised',
            'curr': 'current',
            'prev': 'previous',
            'typ': 'typical',
            'spec': 'special',
            'std': 'standard',
            'cust': 'custom',
            'mod': 'modified',
            'alt': 'alternate',
            'add': 'additional',
            'suppl': 'supplemental',
            'misc': 'miscellaneous',
            'var': 'various',
            'mult': 'multiple',
            'sev': 'several',
            'num': 'numerous',
            'qty': 'quantity',
            'amt': 'amount',
            'no': 'number',
            'pct': 'percent',
            'deg': 'degree',
            'rad': 'radian',
            'lin': 'linear',
            'sq': 'square',
            'cu': 'cubic',
            'rect': 'rectangular',
            'circ': 'circular',
            'tri': 'triangular',
            'hex': 'hexagonal',
            'oct': 'octagonal',
            'irreg': 'irregular',
            'sym': 'symmetrical',
            'asym': 'asymmetrical',
            'uni': 'uniform',
            'var': 'variable',
            'const': 'constant',
            'cont': 'continuous',
            'disc': 'discontinuous',
            'int': 'intermittent',
            'norm': 'normal',
            'abnorm': 'abnormal',
            'reg': 'regular',
            'irreg': 'irregular',
            'smooth': 'smooth',
            'rough': 'rough',
            'flat': 'flat',
            'curved': 'curved',
            'straight': 'straight',
            'bent': 'bent',
            'horiz': 'horizontal',
            'vert': 'vertical',
            'diag': 'diagonal',
            'perp': 'perpendicular',
            'parl': 'parallel',
            'tang': 'tangent',
            'rad': 'radial',
            'concen': 'concentric',
            'eccen': 'eccentric',
            'align': 'aligned',
            'offset': 'offset',
            'cent': 'centered',
            'flush': 'flush',
            'recess': 'recessed',
            'proj': 'projected',
            'embed': 'embedded',
            'surf': 'surface',
            'conc': 'concealed',
            'exp': 'exposed',
            'acc': 'accessible',
            'rem': 'removable',
            'fix': 'fixed',
            'adj': 'adjustable',
            'port': 'portable',
            'perm': 'permanent',
            'temp': 'temporary'
        }
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info("Preprocessor initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        # Pattern for multiple spaces
        self.multi_space_pattern = re.compile(r'\s+')
        
        # Pattern for common punctuation issues
        self.punct_pattern = re.compile(r'([.!?])\1+')
        
        # Pattern for mixed case IDs (e.g., rFi, Cb)
        self.mixed_case_id_pattern = re.compile(r'\b(rfi|cb|ahu|vav|mep|hvac)\b', re.IGNORECASE)
        
        # Pattern for ID with numbers
        self.id_number_pattern = re.compile(r'\b(RFI|CB|submittal|door|room)\s*#?\s*(\d+)', re.IGNORECASE)
        
        # Pattern for CSI sections
        self.csi_pattern = re.compile(r'\b(\d{2})[\s\-\.]*(\d{2})[\s\-\.]*(\d{2})\b')
        self.csi_continuous_pattern = re.compile(r'\b(\d{6})\b')
        
        # Pattern for product codes
        self.product_code_pattern = re.compile(r'\b([A-Z]{1,4})[\s\-]*(\d{1,3})\b')
        
        # Pattern for drawing references
        self.drawing_ref_pattern = re.compile(r'\b([A-Z]+)[\s\-]*(\d+(?:\.\d+)?)\b')
        
        # Pattern for door IDs
        self.door_pattern = re.compile(r'\b(\d{4,5})(?:\s*-\s*(\d))?\b')
        
        # Pattern for floor/level references
        self.floor_pattern = re.compile(r'\b(?:floor|level|flr|lvl|L)\s*(\d+|[A-Z]\d*)\b', re.IGNORECASE)
        
        # Pattern for question marks at the end
        self.question_mark_pattern = re.compile(r'\s*\?\s*$')
        
        # Pattern for units
        self.unit_pattern = re.compile(r'\b(\d+)\s*(sf|lf|cf|sy|cy|gal|lbs?|psi|psf|plf|kips|pcf|mph|fps|fpm|cfm|gpm|btu|btuh|kw|hp|v|a|hz)\b', re.IGNORECASE)
        
        # Pattern for dimensions
        self.dimension_pattern = re.compile(r'\b(\d+)\s*[\'"]?\s*[-x]\s*(\d+)\s*[\'"]?\b')
        
        # Pattern for percentages
        self.percent_pattern = re.compile(r'\b(\d+)\s*(?:%|percent|pct)\b', re.IGNORECASE)
        
        # Pattern for temperatures
        self.temp_pattern = re.compile(r'\b(\d+)\s*(?:°|deg|degrees?)\s*([CF])\b', re.IGNORECASE)
        
        # Pattern for ranges
        self.range_pattern = re.compile(r'\b(\d+)\s*(?:to|-|thru|through)\s*(\d+)\b', re.IGNORECASE)
        
        # Pattern for fractions
        self.fraction_pattern = re.compile(r'\b(\d+)\s*/\s*(\d+)\b')
        
        # Pattern for abbreviations at word boundaries
        self.abbrev_pattern = re.compile(r'\b(' + '|'.join(re.escape(abbr) for abbr in self.abbreviations.keys()) + r')\b', re.IGNORECASE)
    
    def process(self, text: str) -> str:
        """
        Main preprocessing method.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        original_text = text
        
        # Step 1: Unicode normalization
        text = self._normalize_unicode(text)
        
        # Step 2: Basic cleaning
        text = self._basic_clean(text)
        
        # Step 3: Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Step 4: Normalize IDs
        text = self._normalize_ids(text)
        
        # Step 5: Normalize CSI sections
        text = self._normalize_csi_sections(text)
        
        # Step 6: Normalize product codes
        text = self._normalize_product_codes(text)
        
        # Step 7: Normalize units and measurements
        text = self._normalize_units(text)
        
        # Step 8: Apply spelling correction
        if self.enable_spell_correction and self.spelling_corrector:
            text = self.spelling_corrector.correct(text)
        
        # Step 9: Final cleaning
        text = self._final_clean(text)
        
        logger.debug(f"Preprocessed: '{original_text}' -> '{text}'")
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common unicode quotes and dashes
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
            '°': ' degrees ',
            '²': '2',
            '³': '3',
            '×': 'x',
            '÷': '/',
            '±': '+/-',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '≈': '~',
            '•': '-',
            '·': '-',
            '®': '',
            '™': '',
            '©': '',
            '\u00A0': ' ',  # Non-breaking space
            '\u2009': ' ',  # Thin space
            '\u200B': '',   # Zero-width space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        # Replace tabs and newlines with spaces
        text = text.replace('\t', ' ').replace('\n', ' ')
        
        # Remove extra spaces
        text = self.multi_space_pattern.sub(' ', text)
        
        # Remove duplicate punctuation
        text = self.punct_pattern.sub(r'\1', text)
        
        return text.strip()
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common construction abbreviations."""
        def replace_abbrev(match):
            abbrev = match.group(1).lower()
            replacement = self.abbreviations.get(abbrev, match.group(1))
            
            # Preserve original case for certain abbreviations
            if abbrev in ['rfi', 'cb', 'ahu', 'vav', 'mep', 'hvac', 'vfd', 'bms', 'plc', 'ul', 'astm', 'ansi', 'aia', 'leed', 'ada', 'osha', 'voc', 'nfpa', 'ibc', 'imc', 'ipc', 'nec']:
                return replacement.upper()
            
            # Check if original was uppercase
            if match.group(1).isupper():
                return replacement.upper()
            elif match.group(1)[0].isupper():
                return replacement.capitalize()
            else:
                return replacement
        
        text = self.abbrev_pattern.sub(replace_abbrev, text)
        
        return text
    
    def _normalize_ids(self, text: str) -> str:
        """Normalize various ID formats."""
        # Normalize RFI, CB, submittal IDs
        text = normalize_id(text)
        
        # Normalize door IDs (ensure proper format)
        def normalize_door(match):
            door_num = match.group(1)
            suffix = match.group(2)
            if suffix:
                return f"{door_num}-{suffix}"
            return door_num
        
        # Only apply to contexts where it's likely a door ID
        door_context_pattern = re.compile(r'\b(?:door|dr)\s*#?\s*(\d{4,5})(?:\s*-\s*(\d))?\b', re.IGNORECASE)
        
        def door_replace(match):
            door_num = match.group(1)
            suffix = match.group(2)
            if suffix:
                return f"door {door_num}-{suffix}"
            return f"door {door_num}"
        
        text = door_context_pattern.sub(door_replace, text)
        
        return text
    
    def _normalize_csi_sections(self, text: str) -> str:
        """Normalize CSI section numbers."""
        text = normalize_csi_section(text)
        return text
    
    def _normalize_product_codes(self, text: str) -> str:
        """Normalize product codes to consistent format."""
        def normalize_product(match):
            prefix = match.group(1).upper()
            number = match.group(2)
            return f"{prefix}-{number}"
        
        # Only in contexts where it's likely a product code
        product_context_pattern = re.compile(r'\b(ACT|AP|CT|DF|VAV|FCU|AHU|RTU|EF|SF|RF)\s*[-]?\s*(\d{1,3})\b', re.IGNORECASE)
        text = product_context_pattern.sub(normalize_product, text)
        
        return text
    
    def _normalize_units(self, text: str) -> str:
        """Normalize units and measurements."""
        # Normalize unit abbreviations
        unit_replacements = {
            'sf': 'square feet',
            'lf': 'linear feet',
            'cf': 'cubic feet',
            'sy': 'square yards',
            'cy': 'cubic yards',
            'gal': 'gallons',
            'lbs': 'pounds',
            'lb': 'pounds',
            'psi': 'PSI',
            'psf': 'PSF',
            'plf': 'PLF',
            'pcf': 'PCF',
            'mph': 'MPH',
            'fps': 'FPS',
            'fpm': 'FPM',
            'cfm': 'CFM',
            'gpm': 'GPM',
            'btu': 'BTU',
            'btuh': 'BTUH',
            'kw': 'kW',
            'hp': 'HP',
            'v': 'volts',
            'a': 'amps',
            'hz': 'Hz'
        }
        
        def replace_unit(match):
            number = match.group(1)
            unit = match.group(2).lower()
            replacement = unit_replacements.get(unit, unit)
            return f"{number} {replacement}"
        
        text = self.unit_pattern.sub(replace_unit, text)
        
        # Normalize dimensions (e.g., 4x8, 4' x 8')
        text = self.dimension_pattern.sub(r'\1 x \2', text)
        
        # Normalize percentages
        text = re.sub(r'\b(\d+)\s*%', r'\1 percent', text)
        text = re.sub(r'\b(\d+)\s*pct\b', r'\1 percent', text, flags=re.IGNORECASE)
        
        # Normalize temperatures
        def replace_temp(match):
            degrees = match.group(1)
            scale = match.group(2).upper()
            return f"{degrees} degrees {scale}"
        
        text = self.temp_pattern.sub(replace_temp, text)
        
        # Normalize ranges
        text = self.range_pattern.sub(r'\1 to \2', text)
        
        return text
    
    def _final_clean(self, text: str) -> str:
        """Final cleaning pass."""
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s+([.,;!?])', r'\1', text)
        text = re.sub(r'([.,;!?])(?=[A-Za-z])', r'\1 ', text)
        
        # Clean up multiple spaces again
        text = self.multi_space_pattern.sub(' ', text)
        
        # Ensure question marks are preserved
        if not text.strip().endswith('?') and self.question_mark_pattern.search(text):
            text = text.rstrip() + '?'
        
        return text.strip()
    
    def batch_process(self, texts: List[str]) -> List[str]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.process(text) for text in texts]
    
    def get_preprocessing_stats(self, text: str) -> Dict[str, any]:
        """
        Get statistics about preprocessing changes.
        
        Args:
            text: Original text
            
        Returns:
            Dictionary with preprocessing statistics
        """
        processed = self.process(text)
        
        stats = {
            'original_length': len(text),
            'processed_length': len(processed),
            'length_change': len(processed) - len(text),
            'abbreviations_expanded': len(self.abbrev_pattern.findall(text)),
            'ids_found': len(self.id_number_pattern.findall(text)),
            'csi_sections_found': len(self.csi_pattern.findall(text)) + len(self.csi_continuous_pattern.findall(text)),
            'product_codes_found': len(re.findall(r'\b(?:ACT|AP|CT|DF|VAV|FCU|AHU|RTU|EF|SF|RF)\s*[-]?\s*\d{1,3}\b', text, re.IGNORECASE)),
            'units_normalized': len(self.unit_pattern.findall(text)),
            'has_question_mark': text.strip().endswith('?'),
            'original_text': text,
            'processed_text': processed
        }
        
        return stats
