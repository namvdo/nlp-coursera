import numpy as np
def test_sigmoid(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {"name": "default_check", "input": {"z": 0}, "expected": 0.5},
        {
            "name": "positive_check",
            "input": {"z": 4.92},
            "expected": 0.9927537604041685,
        },
        {"name": "negative_check", "input": {"z": -1}, "expected": 0.2689414213699951},
        {
            "name": "larger_neg_check",
            "input": {"z": -20},
            "expected": 2.0611536181902037e-09,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output from sigmoid function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases