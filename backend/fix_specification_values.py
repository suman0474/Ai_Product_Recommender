# fix_specification_values.py
# =============================================================================
# SPECIFICATION VALUE CLEANUP UTILITY
# =============================================================================
#
# This script identifies and fixes specification values that contain generic
# text, procedural requirements, or equipment lists instead of technical values.
#
# Usage:
#   python fix_specification_values.py --input specs.json --output fixed_specs.json
#   python fix_specification_values.py --validate specs.json  # Just validate, don't fix
#
# =============================================================================

import sys
import os
import json
import argparse
from typing import Dict, Any, List

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from agentic.deep_agent.spec_value_validator import (
    validate_specifications,
    get_validation_summary,
    print_validation_report,
    separate_specs_by_source,
    filter_valid_specs,
    SourceTag
)


def load_specifications(filepath: str) -> Dict[str, Any]:
    """Load specifications from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_specifications(specs: Dict[str, Any], filepath: str):
    """Save specifications to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(specs, f, indent=2, ensure_ascii=False)


def restructure_specifications(specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restructure specifications by separating technical specs from procedures.
    
    Returns:
        {
            "technical_specifications": {...},
            "maintenance_procedures": {...},
            "safety_requirements": {...},
            "selection_criteria": {...},
            "invalid_removed": [...]
        }
    """
    separated = separate_specs_by_source(specs)
    
    # Add source tags to each section
    result = {
        "technical_specifications": {},
        "maintenance_procedures": {},
        "safety_requirements": {},
        "requirements_and_criteria": {},
        "invalid_items_removed": list(separated.get("invalid", {}).keys())
    }
    
    # Process technical specs - determine appropriate source tag
    for key, value in separated["technical_specs"].items():
        # Add source tag based on field name
        key_lower = key.lower()
        if 'calibrat' in key_lower:
            tag = SourceTag.CALIBRATION.value
        elif any(x in key_lower for x in ['set', 'configured', 'programmed']):
            tag = SourceTag.CONFIGURATION.value
        elif any(x in key_lower for x in ['model', 'serial', 'manufacturer']):
            tag = SourceTag.NAMEPLATE.value
        else:
            tag = SourceTag.DATASHEET.value
        
        result["technical_specifications"][key] = {
            "value": value,
            "source": tag
        }
    
    # Process procedures
    for key, value in separated["procedures"].items():
        if 'safety' in key.lower():
            result["safety_requirements"][key] = value
        else:
            result["maintenance_procedures"][key] = value
    
    # Process requirements
    for key, value in separated["requirements"].items():
        result["requirements_and_criteria"][key] = value
    
    # Remove empty sections
    result = {k: v for k, v in result.items() if v}
    
    return result


def generate_report(specs: Dict[str, Any], product_type: str = "Product") -> str:
    """Generate a detailed text report of specification issues."""
    validation_results = validate_specifications(specs, product_type)
    summary = get_validation_summary(validation_results)
    separated = separate_specs_by_source(specs, product_type)
    
    report = []
    report.append("=" * 80)
    report.append(f"SPECIFICATION VALIDATION REPORT: {product_type}")
    report.append("=" * 80)
    report.append("")
    
    # Summary
    report.append("SUMMARY:")
    report.append(f"  Total Specifications: {summary['total_specs']}")
    report.append(f"  Valid Technical Specs: {summary['valid_specs']} ({summary['validity_rate']:.1%})")
    report.append(f"  Invalid/Procedural: {summary['invalid_specs']} ({1-summary['validity_rate']:.1%})")
    report.append("")
    
    # Breakdown by type
    report.append("BREAKDOWN BY TYPE:")
    for type_name, count in sorted(summary['by_type'].items()):
        report.append(f"  {type_name}: {count}")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDED ACTIONS:")
    report.append("")
    
    # Technical specs that are valid
    if separated["technical_specs"]:
        report.append(f"✓ KEEP AS TECHNICAL SPECIFICATIONS ({len(separated['technical_specs'])} items):")
        for key in sorted(separated["technical_specs"].keys())[:10]:  # Show first 10
            value = str(separated["technical_specs"][key])[:60]
            report.append(f"  • {key}: {value}")
        if len(separated["technical_specs"]) > 10:
            report.append(f"  ... and {len(separated['technical_specs']) - 10} more")
        report.append("")
    
    # Procedures
    if separated["procedures"]:
        report.append(f"→ MOVE TO 'PROCEDURES' SECTION ({len(separated['procedures'])} items):")
        for key in sorted(separated["procedures"].keys())[:5]:
            value = str(separated["procedures"][key])[:60]
            report.append(f"  • {key}: {value}")
        if len(separated["procedures"]) > 5:
            report.append(f"  ... and {len(separated['procedures']) - 5} more")
        report.append("")
    
    # Requirements
    if separated["requirements"]:
        report.append(f"→ MOVE TO 'REQUIREMENTS' SECTION ({len(separated['requirements'])} items):")
        for key in sorted(separated["requirements"].keys())[:5]:
            value = str(separated["requirements"][key])[:60]
            report.append(f"  • {key}: {value}")
        if len(separated["requirements"]) > 5:
            report.append(f"  ... and {len(separated['requirements']) - 5} more")
        report.append("")
    
    # Invalid items
    if separated["invalid"]:
        report.append(f"✗ ITEMS TO REMOVE OR CLARIFY ({len(separated['invalid'])} items):")
        for key, value in list(separated["invalid"].items())[:5]:
            val_str = str(value)[:60]
            result = validation_results.get(key)
            reason = result.reason if result else "Unknown"
            report.append(f"  • {key}: {val_str}")
            report.append(f"    Reason: {reason}")
        if len(separated["invalid"]) > 5:
            report.append(f"  ... and {len(separated['invalid']) - 5} more")
        report.append("")
    
    report.append("=" * 80)
    report.append("")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Validate and fix specification values"
    )
    parser.add_argument(
        '--input', '-i',
        help='Input JSON file with specifications',
        required=False
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file for fixed specifications',
        required=False
    )
    parser.add_argument(
        '--product-type', '-p',
        help='Product type (e.g., "Pressure_Relief_Valve")',
        default="Product"
    )
    parser.add_argument(
        '--validate-only', '-v',
        action='store_true',
        help='Only validate, do not fix'
    )
    parser.add_argument(
        '--report', '-r',
        help='Output text file for validation report'
    )
    parser.add_argument(
        '--filter-only', '-f',
        action='store_true',
        help='Filter to keep only valid specs (remove invalid ones)'
    )
    parser.add_argument(
        '--restructure', '-s',
        action='store_true',
        help='Restructure into separate sections (technical/procedures/requirements)'
    )
    
    args = parser.parse_args()
    
    # Load specifications
    if args.input:
        print(f"Loading specifications from: {args.input}")
        specs = load_specifications(args.input)
    else:
        # Use example from command line or predefined example
        print("No input file specified. Use --input to specify a JSON file.")
        print("Example: python fix_specification_values.py --input specs.json --validate-only")
        return
    
    # Validate
    print(f"\nValidating {len(specs)} specifications for {args.product_type}...")
    print("=" * 80)
    print_validation_report(specs, args.product_type)
    
    # Generate text report if requested
    if args.report:
        report_text = generate_report(specs, args.product_type)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\nValidation report saved to: {args.report}")
    
    # Stop here if validate-only
    if args.validate_only:
        return
    
    # Fix specifications
    if args.output:
        if args.filter_only:
            # Filter mode: Keep only valid specs
            print(f"\nFiltering to keep only valid technical specifications...")
            fixed_specs = filter_valid_specs(specs, args.product_type)
            print(f"Retained {len(fixed_specs)} valid specifications")
            
        elif args.restructure:
            # Restructure mode: Separate into sections
            print(f"\nRestructuring specifications into separate sections...")
            fixed_specs = restructure_specifications(specs)
            print(f"Structured into {len(fixed_specs)} sections")
            for section, items in fixed_specs.items():
                if isinstance(items, dict):
                    print(f"  {section}: {len(items)} items")
                elif isinstance(items, list):
                    print(f"  {section}: {len(items)} items")
        else:
            # Default: Just remove invalid
            print(f"\nRemoving invalid specifications...")
            fixed_specs = filter_valid_specs(specs, args.product_type)
        
        # Save
        save_specifications(fixed_specs, args.output)
        print(f"\nFixed specifications saved to: {args.output}")
        
        # Show summary
        print(f"\nSUMMARY:")
        print(f"  Original: {len(specs)} specifications")
        if isinstance(fixed_specs, dict):
            if 'technical_specifications' in fixed_specs:
                total_fixed = sum(len(v) if isinstance(v, (dict, list)) else 1 
                                  for v in fixed_specs.values())
            else:
                total_fixed = len(fixed_specs)
        else:
            total_fixed = len(fixed_specs)
        print(f"  Fixed: {total_fixed} items")
        print(f"  Removed: {len(specs) - total_fixed} invalid items")


if __name__ == "__main__":
    main()
