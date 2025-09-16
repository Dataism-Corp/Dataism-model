import os
import sys
import json
import hashlib
import logging
import argparse
import sqlite3
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any


class VerificationTool:
    def __init__(self, phase: int, evidence_dir: str = "./evidence", simulate_fail: str = None):
        self.phase = phase
        self.evidence_dir = Path(evidence_dir)
        self.simulate_fail = simulate_fail
        self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        self.run_dir = self.evidence_dir / f"phase{phase}" / self.timestamp
        self.artifacts_dir = self.run_dir / "artifacts"
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        self.results = []
        self.start_time = datetime.now()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging to both console and file"""
        logger = logging.getLogger('verification')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # File handler
        log_file = self.run_dir / "verify.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _save_artifact(self, name: str, content: Any, artifact_type: str = "text") -> str:
        """Save an artifact and return its path"""
        if artifact_type == "json":
            file_path = self.artifacts_dir / f"{name}.json"
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        else:
            file_path = self.artifacts_dir / f"{name}.txt"
            with open(file_path, 'w') as f:
                f.write(str(content))
        
        return str(file_path.relative_to(self.run_dir))

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def check_file_structure(self) -> Tuple[bool, str, Dict]:
        """Check if required file structure exists"""
        check_name = "file_structure"
        self.logger.info(f"Running check: {check_name}")
        
        # Simulate failure if requested
        if self.simulate_fail == check_name:
            self.logger.error(f"Simulated failure for {check_name}")
            return False, "Simulated failure: file_structure", {}
        
        required_files = [
            "test_files/config.json",
            "test_files/data.db",
            "test_files/requirements.txt"
        ]
        
        results = {}
        all_exist = True
        
        for file_path in required_files:
            exists = os.path.exists(file_path)
            results[file_path] = {
                "exists": exists,
                "size": os.path.getsize(file_path) if exists else 0,
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat() if exists else None
            }
            if not exists:
                all_exist = False
                self.logger.error(f"Missing required file: {file_path}")
            else:
                self.logger.info(f"Found file: {file_path} ({results[file_path]['size']} bytes)")
        
        # Save artifact
        artifact_path = self._save_artifact("file_structure_check", results, "json")
        
        status = "PASS" if all_exist else "FAIL"
        message = f"File structure check: {len([r for r in results.values() if r['exists']])}/{len(required_files)} files found"
        
        return all_exist, message, {"artifact": artifact_path, "details": results}

    def check_database_schema(self) -> Tuple[bool, str, Dict]:
        """Check database schema and contents"""
        check_name = "database_schema"
        self.logger.info(f"Running check: {check_name}")
        
        # Simulate failure if requested
        if self.simulate_fail == check_name:
            self.logger.error(f"Simulated failure for {check_name}")
            return False, "Simulated failure: database_schema", {}
        
        db_path = "test_files/data.db"
        if not os.path.exists(db_path):
            return False, "Database file not found", {}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            results = {"tables": {}}
            
            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                
                results["tables"][table_name] = {
                    "columns": [{"name": col[1], "type": col[2], "nullable": not col[3]} for col in columns],
                    "row_count": row_count
                }
                self.logger.info(f"Table {table_name}: {len(columns)} columns, {row_count} rows")
            
            conn.close()
            
            # Save artifact
            artifact_path = self._save_artifact("database_schema", results, "json")
            
            expected_tables = ["users", "projects"]
            found_tables = [t[0] for t in tables]
            has_required_tables = all(table in found_tables for table in expected_tables)
            
            status = "PASS" if has_required_tables else "FAIL"
            message = f"Database schema: {len(found_tables)} tables found, required tables: {has_required_tables}"
            
            return has_required_tables, message, {"artifact": artifact_path, "details": results}
            
        except Exception as e:
            self.logger.error(f"Database check failed: {str(e)}")
            return False, f"Database error: {str(e)}", {}

    def check_api_endpoint(self) -> Tuple[bool, str, Dict]:
        """Check API endpoint availability"""
        check_name = "api_endpoint"
        self.logger.info(f"Running check: {check_name}")
        
        # Simulate failure if requested
        if self.simulate_fail == check_name:
            self.logger.error(f"Simulated failure for {check_name}")
            return False, "Simulated failure: api_endpoint", {}
        
        # For testing purposes, we'll check a local mock endpoint or httpbin
        test_endpoints = [
            "https://httpbin.org/status/200",  # Should return 200
            "https://httpbin.org/json",        # Should return JSON
        ]
        
        results = {}
        all_passed = True
        
        for url in test_endpoints:
            try:
                self.logger.info(f"Testing endpoint: {url}")
                response = requests.get(url, timeout=10)
                
                results[url] = {
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "headers": dict(response.headers),
                    "success": 200 <= response.status_code < 300
                }
                
                if results[url]["success"]:
                    self.logger.info(f"Endpoint {url}: OK ({response.status_code})")
                else:
                    self.logger.error(f"Endpoint {url}: FAIL ({response.status_code})")
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Endpoint {url}: ERROR - {str(e)}")
                results[url] = {
                    "error": str(e),
                    "success": False
                }
                all_passed = False
        
        # Save artifact
        artifact_path = self._save_artifact("api_endpoint_check", results, "json")
        
        successful_endpoints = sum(1 for r in results.values() if r.get("success", False))
        message = f"API endpoints: {successful_endpoints}/{len(test_endpoints)} accessible"
        
        return all_passed, message, {"artifact": artifact_path, "details": results}

    def run_verification(self) -> int:
        """Run all verification checks"""
        self.logger.info(f"Starting Phase {self.phase} verification")
        self.logger.info(f"Evidence directory: {self.run_dir}")
        
        # Define checks to run
        checks = [
            ("file_structure", self.check_file_structure),
            ("database_schema", self.check_database_schema),
            ("api_endpoint", self.check_api_endpoint),
        ]
        
        total_checks = len(checks)
        passed_checks = 0
        
        # Run each check
        for check_name, check_func in checks:
            try:
                success, message, evidence = check_func()
                
                result = {
                    "check": check_name,
                    "status": "PASS" if success else "FAIL",
                    "message": message,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "evidence": evidence
                }
                
                self.results.append(result)
                
                if success:
                    passed_checks += 1
                    self.logger.info(f"✅ {check_name}: {message}")
                else:
                    self.logger.error(f"❌ {check_name}: {message}")
                    
            except Exception as e:
                self.logger.error(f"Exception in {check_name}: {str(e)}")
                self.results.append({
                    "check": check_name,
                    "status": "ERROR",
                    "message": f"Exception: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "evidence": {}
                })
        
        # Generate final report
        duration = datetime.now() - self.start_time
        
        report = {
            "phase": self.phase,
            "timestamp": self.timestamp,
            "duration_seconds": duration.total_seconds(),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "overall_status": "PASS" if passed_checks == total_checks else "FAIL",
            "checks": self.results
        }
        
        # Save report
        report_file = self.run_dir / "report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate manifest with file hashes
        manifest = self._generate_manifest()
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Phase {self.phase} Verification Complete")
        print(f"{'='*50}")
        print(f"Status: {report['overall_status']}")
        print(f"Checks: {passed_checks}/{total_checks}")
        print(f"Duration: {duration.total_seconds():.1f}s")
        print(f"Evidence: {self.run_dir}")
        print(f"{'='*50}")
        
        for result in self.results:
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_emoji} {result['check']}: {result['message']}")
        
        print(f"{'='*50}")
        
        # Return appropriate exit code
        return 0 if passed_checks == total_checks else 1

    def _generate_manifest(self) -> Dict:
        """Generate manifest with file hashes"""
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            "files": {}
        }
        
        # Hash all files in the evidence directory
        for file_path in self.run_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.run_dir)
                manifest["files"][str(relative_path)] = self._calculate_hash(file_path)
        
        # Save manifest
        manifest_file = self.run_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest


def main():
    parser = argparse.ArgumentParser(description="Verification Tool for AI System Phases")
    parser.add_argument("--phase", type=int, required=True, help="Phase number to verify")
    parser.add_argument("--evidence-out", default="./evidence", help="Evidence output directory")
    parser.add_argument("--simulate-fail", choices=["file_structure", "database_schema", "api_endpoint"], 
                       help="Simulate failure for specific check")
    
    args = parser.parse_args()
    
    # Create and run verification tool
    verifier = VerificationTool(
        phase=args.phase,
        evidence_dir=args.evidence_out,
        simulate_fail=args.simulate_fail
    )
    
    exit_code = verifier.run_verification()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()