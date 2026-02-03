# test_spec_validation.py
# =============================================================================
# TEST SCRIPT: Validate Pressure Relief Valve Specifications
# =============================================================================

import sys
import os
import io

# Fix Unicode encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agentic.deep_agent.spec_value_validator import (
    validate_specifications,
    print_validation_report,
    separate_specs_by_source,
    filter_valid_specs
)

# Example specifications from user (Pressure Relief Valve)
PRESSURE_RELIEF_VALVE_SPECS = {
    "Adjustment_Parameters": "zero, span, linearity",
    "Cable_Specification": "shielded twisted-pair for analog and digital signals, follow manufacturer's cable requirements for Fieldbus, maintain overall shield continuity",
    "Calibration_Equipment_Accuracy_Traceability": "NIST-traceable accuracy for pressure calibrators",
    "Calibration_Equipment_Maintenance": "calibrate and certify regularly, store in protective cases, charge batteries, update firmware",
    "Calibration_Equipment_Types": "deadweight testers, pressure pumps, electronic calibrators",
    "Calibration_Frequency": "at least annually for safety-critical instruments",
    "Calibration_Points": "multiple points across operating range (e.g., 0, 25, 50, 75, 100%)",
    "Communication_Protocol_Support": "HART, FOUNDATION Fieldbus H1, Profibus PA/DP, Modbus RTU/TCP, WirelessHART / ISA100, Industrial Ethernet",
    "Corrective_Maintenance_Action": "repair or replace failed instruments promptly",
    "Data_Sheet_Fields": "device tag, communication protocol, node address, power consumption, hazardous area certification, input/output type, signal range, isolation rating, manufacturer model",
    "Documentation_Requirements": "calibration data, date, technician, calibration equipment used, certificates",
    "Hazardous_Area_Work_Requirements": "work permits and gas testing",
    "Hysteresis_Minimization": "yes",
    "Intrinsic_Safety_Barrier_Selection_Criteria": "gas group, temperature class, voltage/current requirements, matching with field device entity parameters",
    "Management_System": "CMMS or AMS for planning and tracking",
    "Network_Architecture_Considerations": "segment length limits, number of devices per segment, redundant media for critical systems, isolation with repeaters",
    "Out_Of_Tolerance_Action": "remove from service, tag, repair or replace, evaluate process impact, implement corrective action",
    "Ppe_Requirement": "appropriate PPE when working on process equipment",
    "Pre_Calibration_Inspection": "instrument condition, correct installation, absence of damage or leaks",
    "Predictive_Maintenance_Inputs": "smart diagnostics from transmitters, analyzers, control valves (sensor drift, valve signature, plugged impulse lines)",
    "Predictive_Maintenance_Methods": "data analytics, artificial intelligence",
    "Preventive_Maintenance_Tasks": "routine inspections (visual check of cables, enclosures, indications), clean sensors, replace consumables (filters, membranes, diaphragms), lubricate moving parts",
    "Record_Retention": "maintain calibration certificates and maintenance records for audit and compliance",
    "Safety_Procedure_Before_Disconnection": "vent and depressurize impulse lines",
    "Safety_Procedure_Before_Removal": "lockout/tagout",
    "Signal_Conditioning_Requirements": "Intrinsic safety (IS) barriers, signal isolators, surge protection devices (SPDs), IS power supplies",
    "Signal_Isolation_Criteria": "galvanic isolation for analog signals crossing different ground potentials, appropriate accuracy and response time",
    "Spare_Parts_Inventory": "maintain list of critical spares for regulators and fittings",
    "Spare_Parts_Storage": "controlled environment",
    "Technician_Qualification": "qualified technicians trained on specific instrument types",
    "Training_Topics": "new technologies, safety regulations, maintenance software",
}


# Correct example specifications (technical values)
CORRECT_EXAMPLES = {
    "Set_Pressure": "150 psig ±3%",
    "Orifice_Size": "1.5 inch",
    "Spring_Range": "100-200 psi",
    "Material_Body": "Carbon Steel",
    "Material_Trim": "Stainless Steel 316",
    "Connection_Size_Inlet": "4 inch 300# RF",
    "Connection_Size_Outlet": "6 inch 150# RF",
    "Temperature_Rating": "-20°F to 400°F",
    "Calibration_As_Found": "152 psig",
    "Calibration_As_Left": "150 psig",
    "Calibration_Date": "2024-01-15",
    "Manufacturer": "Anderson Greenwood",
    "Model": "Type 80 Series",
}


def main():
    print("\n" + "="*80)
    print("PRESSURE RELIEF VALVE SPECIFICATION VALIDATION TEST")
    print("="*80)
    
    # Test 1: Validate problematic specifications
    print("\n\n### TEST 1: Analyzing Problematic Specifications ###\n")
    print_validation_report(PRESSURE_RELIEF_VALVE_SPECS, "Pressure Relief Valve")
    
    # Test 2: Show categorization
    print("\n### TEST 2: Recommended Categorization ###\n")
    separated = separate_specs_by_source(PRESSURE_RELIEF_VALVE_SPECS, "Pressure Relief Valve")
    
    for section, specs in separated.items():
        if specs:
            print(f"\n{section.upper().replace('_', ' ')} ({len(specs)} items):")
            for key in sorted(specs.keys()):
                value_preview = str(specs[key])[:80]
                print(f"  • {key}: {value_preview}...")
    
    # Test 3: Filter to keep only valid specs
    print("\n\n### TEST 3: Filtered Valid Specifications ###\n")
    valid_only = filter_valid_specs(PRESSURE_RELIEF_VALVE_SPECS, "Pressure Relief Valve")
    print(f"Valid specifications retained: {len(valid_only)} out of {len(PRESSURE_RELIEF_VALVE_SPECS)}")
    for key, value in valid_only.items():
        print(f"  • {key}: {value}")
    
    # Test 4: Compare with correct examples
    print("\n\n### TEST 4: Correct Specification Examples ###\n")
    print_validation_report(CORRECT_EXAMPLES, "Pressure Relief Valve")
    
    # Summary statistics
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    problematic_validation = validate_specifications(PRESSURE_RELIEF_VALVE_SPECS)
    correct_validation = validate_specifications(CORRECT_EXAMPLES)
    
    prob_valid = sum(1 for r in problematic_validation.values() if r.is_valid)
    correct_valid = sum(1 for r in correct_validation.values() if r.is_valid)
    
    print(f"\nProblematic Set: {prob_valid}/{len(PRESSURE_RELIEF_VALVE_SPECS)} valid ({prob_valid/len(PRESSURE_RELIEF_VALVE_SPECS):.1%})")
    print(f"Correct Examples: {correct_valid}/{len(CORRECT_EXAMPLES)} valid ({correct_valid/len(CORRECT_EXAMPLES):.1%})")
    
    print("\n✅ Validation complete!\n")


if __name__ == "__main__":
    main()
