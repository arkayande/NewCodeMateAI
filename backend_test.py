import requests
import sys
import json
import time
from datetime import datetime

class CodeGuardianAPITester:
    def __init__(self, base_url="https://codeguardian-2.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.analysis_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout}s")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test root API endpoint"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )
        return success

    def test_get_all_analyses(self):
        """Test getting all analyses"""
        success, response = self.run_test(
            "Get All Analyses",
            "GET", 
            "analyses",
            200
        )
        if success:
            print(f"   Found {len(response)} existing analyses")
        return success, response

    def test_start_analysis(self):
        """Test starting a new analysis"""
        test_repo = "https://github.com/octocat/Hello-World.git"
        success, response = self.run_test(
            "Start Analysis",
            "POST",
            "analyze",
            200,
            data={"git_url": test_repo}
        )
        if success and 'id' in response:
            self.analysis_id = response['id']
            print(f"   Analysis ID: {self.analysis_id}")
            print(f"   Status: {response.get('status', 'unknown')}")
        return success, response

    def test_get_specific_analysis(self):
        """Test getting a specific analysis"""
        if not self.analysis_id:
            print("❌ Skipped - No analysis ID available")
            return False
            
        success, response = self.run_test(
            "Get Specific Analysis",
            "GET",
            f"analysis/{self.analysis_id}",
            200
        )
        if success:
            print(f"   Analysis Status: {response.get('status', 'unknown')}")
            print(f"   Repo Name: {response.get('repo_name', 'unknown')}")
        return success

    def test_delete_analysis(self):
        """Test deleting an analysis"""
        if not self.analysis_id:
            print("❌ Skipped - No analysis ID available")
            return False
            
        success, response = self.run_test(
            "Delete Analysis",
            "DELETE",
            f"analysis/{self.analysis_id}",
            200
        )
        if success:
            print(f"   Deletion successful")
        return success

    def test_delete_nonexistent_analysis(self):
        """Test deleting a non-existent analysis"""
        fake_id = "non-existent-id-12345"
        success, response = self.run_test(
            "Delete Non-existent Analysis",
            "DELETE",
            f"analysis/{fake_id}",
            404
        )
        return success

    def test_get_nonexistent_analysis(self):
        """Test getting a non-existent analysis"""
        fake_id = "non-existent-id-12345"
        success, response = self.run_test(
            "Get Non-existent Analysis",
            "GET",
            f"analysis/{fake_id}",
            404
        )
        return success

def main():
    print("🚀 Starting CodeGuardian AI Backend API Tests")
    print("=" * 60)
    
    tester = CodeGuardianAPITester()
    
    # Test sequence
    tests = [
        ("Root Endpoint", tester.test_root_endpoint),
        ("Get All Analyses", lambda: tester.test_get_all_analyses()[0]),
        ("Start Analysis", lambda: tester.test_start_analysis()[0]),
        ("Get Specific Analysis", tester.test_get_specific_analysis),
        ("Delete Analysis", tester.test_delete_analysis),
        ("Delete Non-existent Analysis", tester.test_delete_nonexistent_analysis),
        ("Get Non-existent Analysis", tester.test_get_nonexistent_analysis),
    ]
    
    # Run tests
    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            tester.tests_run += 1
        
        # Small delay between tests
        time.sleep(1)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {tester.tests_run - tester.tests_passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())