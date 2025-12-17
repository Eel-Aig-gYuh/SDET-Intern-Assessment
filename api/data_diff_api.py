from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse
import pandas as pd
import numpy as np
import io
from typing import Dict
from datetime import datetime


app = FastAPI(title="CSV Comparison Service")


class CSVComparator:
    def __init__(
            self,
            df1: pd.DataFrame,
            df2: pd.DataFrame,
            primary_key: str,
            numeric_tolerance: float = 1e-9,
            max_mismatches: int = 10000
    ):
        """
            Hàm khởi tạo
            :param df1: Dataframe từ CSV thứ nhất
            :param df2: Dataframe từ CSV thứ hai
            :param primary_key: cột dùng để so sánh
        """
        self.df1 = df1
        self.df2 = df2
        self.primary_key = primary_key
        self.numeric_tolerance = numeric_tolerance
        self.max_mismatches = max_mismatches
        self.results = {
            "missing_in_df2": [],
            "missing_in_df1": [],
            "duplicate_keys_df1": [],
            "duplicate_keys_df2": [],
            "columns_only_in_df1": [],
            "columns_only_in_df2": [],
            "field_mismatches": [],
            "total_compared": 0,
            "quality_score": 0.0,
            "comparison_summary": {}
        }

    def _validate_and_prepare(self):
        """Validate data và chuẩn bị cho việc so sánh"""
        # Kiểm tra primary key tồn tại
        if self.primary_key not in self.df1.columns:
            raise ValueError(f"Primary key '{self.primary_key}' không tồn tại trong File 1")
        if self.primary_key not in self.df2.columns:
            raise ValueError(f"Primary key '{self.primary_key}' không tồn tại trong File 2")

        # Kiểm tra duplicate keys
        dup1 = self.df1[self.df1.duplicated(subset=[self.primary_key], keep=False)]
        dup2 = self.df2[self.df2.duplicated(subset=[self.primary_key], keep=False)]

        if not dup1.empty:
            self.results["duplicate_keys_df1"] = dup1[self.primary_key].unique().tolist()
        if not dup2.empty:
            self.results["duplicate_keys_df2"] = dup2[self.primary_key].unique().tolist()

        # Strip whitespace từ column names
        self.df1.columns = self.df1.columns.str.strip()
        self.df2.columns = self.df2.columns.str.strip()

        # Strip whitespace từ primary key values nếu là string
        if self.df1[self.primary_key].dtype == 'object':
            self.df1[self.primary_key] = self.df1[self.primary_key].astype(str).str.strip()
        if self.df2[self.primary_key].dtype == 'object':
            self.df2[self.primary_key] = self.df2[self.primary_key].astype(str).str.strip()

        # Set primary key làm index (loại bỏ duplicates, giữ first)
        self.df1 = self.df1.drop_duplicates(subset=[self.primary_key]).set_index(self.primary_key)
        self.df2 = self.df2.drop_duplicates(subset=[self.primary_key]).set_index(self.primary_key)

    def _find_missing_keys(self):
        """Tìm keys bị thiếu giữa 2 files - O(n) time complexity"""
        keys_df1 = set(self.df1.index)
        keys_df2 = set(self.df2.index)

        missing_in_df2 = keys_df1 - keys_df2
        missing_in_df1 = keys_df2 - keys_df1
        common_keys = keys_df1 & keys_df2

        self.results["missing_in_df2"] = sorted(list(missing_in_df2))
        self.results["missing_in_df1"] = sorted(list(missing_in_df1))
        self.results["total_compared"] = len(common_keys)

        return common_keys

    def _find_column_differences(self):
        """Tìm các columns chỉ có trong 1 file"""
        cols1 = set(self.df1.columns)
        cols2 = set(self.df2.columns)

        self.results["columns_only_in_df1"] = sorted(list(cols1 - cols2))
        self.results["columns_only_in_df2"] = sorted(list(cols2 - cols1))

        return cols1 & cols2  # Common columns

    def _compare_values_vectorized(self, common_keys, common_cols):
        """
        So sánh values sử dụng vectorization - OPTIMIZED
        Thay vì loop từng row, sử dụng pandas operations
        """
        # Lọc data chỉ lấy common keys và columns
        df1_common = self.df1.loc[list(common_keys), list(common_cols)]
        df2_common = self.df2.loc[list(common_keys), list(common_cols)]

        mismatches = []
        mismatch_count = 0

        # Duyệt qua từng column
        for col in common_cols:
            # Lấy series cho column
            s1 = df1_common[col]
            s2 = df2_common[col]

            # Tìm differences
            # Case 1: Cả 2 đều NaN -> không phải difference
            both_nan = s1.isna() & s2.isna()

            # Case 2: Một bên NaN, bên kia không -> difference
            one_nan = s1.isna() ^ s2.isna()

            # Case 3: Cả 2 đều không NaN -> cần so sánh value
            both_not_nan = ~s1.isna() & ~s2.isna()

            # So sánh values với tolerance cho numeric
            if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
                # Numeric comparison với tolerance
                value_diff = both_not_nan & ~np.isclose(
                    s1.astype(float),
                    s2.astype(float),
                    rtol=self.numeric_tolerance,
                    equal_nan=False
                )
            else:
                # String/Object comparison
                value_diff = both_not_nan & (s1.astype(str) != s2.astype(str))

            # Tất cả differences cho column này
            all_diffs = one_nan | value_diff

            # Lấy keys có difference
            diff_keys = all_diffs[all_diffs].index.tolist()

            # Tạo mismatch records
            for key in diff_keys:
                if mismatch_count >= self.max_mismatches:
                    break

                val1 = s1.loc[key]
                val2 = s2.loc[key]

                # Tìm existing mismatch record cho key này
                existing = next((m for m in mismatches if m["primary_key"] == str(key)), None)

                field_mismatch = {
                    "field": col,
                    "value_file1": str(val1) if not pd.isna(val1) else "NULL",
                    "value_file2": str(val2) if not pd.isna(val2) else "NULL"
                }

                if existing:
                    existing["mismatched_fields"].append(field_mismatch)
                else:
                    mismatches.append({
                        "primary_key": str(key),
                        "mismatched_fields": [field_mismatch]
                    })

                mismatch_count += 1

            if mismatch_count >= self.max_mismatches:
                break

        self.results["field_mismatches"] = mismatches
        self.results["comparison_summary"]["total_field_mismatches"] = mismatch_count
        self.results["comparison_summary"]["truncated"] = mismatch_count >= self.max_mismatches

    def compare(self) -> Dict:
        """
        Main comparison logic - OPTIMIZED
        Time complexity: O(n*m) where n=rows, m=columns (vectorized operations)
        """
        # Step 1: Validate và prepare data
        self._validate_and_prepare()

        # Step 2: Tìm missing keys - O(n)
        common_keys = self._find_missing_keys()

        # Step 3: Tìm column differences - O(m)
        common_cols = self._find_column_differences()

        # Step 4: Compare values sử dụng vectorization - O(n*m)
        if common_keys and common_cols:
            self._compare_values_vectorized(common_keys, common_cols)

        # Step 5: Calculate quality score
        self.results["quality_score"] = self._calculate_quality_score()

        return self.results

    def _calculate_quality_score(self) -> float:
        """
        Tính quality score dựa trên nhiều factors
        """
        total_rows = len(self.df1) + len(self.df2)
        if total_rows == 0:
            return 100.0

        # Weight các loại issues khác nhau
        issues_score = 0

        # Missing rows (weight: 1.0)
        missing_count = len(self.results["missing_in_df1"]) + len(self.results["missing_in_df2"])
        issues_score += missing_count * 1.0

        # Field mismatches (weight: 0.5)
        field_mismatch_count = sum(
            len(m["mismatched_fields"])
            for m in self.results["field_mismatches"]
        )
        issues_score += field_mismatch_count * 0.5

        # Duplicate keys (weight: 0.8)
        dup_count = len(self.results["duplicate_keys_df1"]) + len(self.results["duplicate_keys_df2"])
        issues_score += dup_count * 0.8

        # Column mismatches (weight: 0.3)
        col_mismatch = len(self.results["columns_only_in_df1"]) + len(self.results["columns_only_in_df2"])
        issues_score += col_mismatch * 0.3

        # Calculate score
        score = max(0, 100 - (issues_score / total_rows * 100))
        return round(score, 2)

    def generate_report_csv(self) -> str:
        """Generate detailed comparison report"""
        output = io.StringIO()

        output.write("=== CSV COMPARISON REPORT (OPTIMIZED) ===\n")
        output.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output.write(f"Primary Key: {self.primary_key}\n")
        output.write(f"Numeric Tolerance: {self.numeric_tolerance}\n\n")

        output.write("=== SUMMARY ===\n")
        output.write(f"Total rows in File 1,{len(self.df1)}\n")
        output.write(f"Total rows in File 2,{len(self.df2)}\n")
        output.write(f"Duplicate keys in File 1,{len(self.results['duplicate_keys_df1'])}\n")
        output.write(f"Duplicate keys in File 2,{len(self.results['duplicate_keys_df2'])}\n")
        output.write(f"Missing in File 2,{len(self.results['missing_in_df2'])}\n")
        output.write(f"Missing in File 1,{len(self.results['missing_in_df1'])}\n")
        output.write(f"Rows compared,{self.results['total_compared']}\n")
        output.write(f"Field mismatches,{len(self.results['field_mismatches'])}\n")
        output.write(f"Columns only in File 1,{len(self.results['columns_only_in_df1'])}\n")
        output.write(f"Columns only in File 2,{len(self.results['columns_only_in_df2'])}\n")
        output.write(f"Data Quality Score,{self.results['quality_score']}%\n\n")

        # Duplicate keys
        if self.results["duplicate_keys_df1"]:
            output.write("=== DUPLICATE KEYS IN FILE 1 ===\n")
            output.write(f"{self.primary_key}\n")
            for key in self.results["duplicate_keys_df1"]:
                output.write(f"{key}\n")
            output.write("\n")

        if self.results["duplicate_keys_df2"]:
            output.write("=== DUPLICATE KEYS IN FILE 2 ===\n")
            output.write(f"{self.primary_key}\n")
            for key in self.results["duplicate_keys_df2"]:
                output.write(f"{key}\n")
            output.write("\n")

        # Column differences
        if self.results["columns_only_in_df1"]:
            output.write("=== COLUMNS ONLY IN FILE 1 ===\n")
            for col in self.results["columns_only_in_df1"]:
                output.write(f"{col}\n")
            output.write("\n")

        if self.results["columns_only_in_df2"]:
            output.write("=== COLUMNS ONLY IN FILE 2 ===\n")
            for col in self.results["columns_only_in_df2"]:
                output.write(f"{col}\n")
            output.write("\n")

        # Missing rows
        if self.results["missing_in_df2"]:
            output.write("=== MISSING IN FILE 2 ===\n")
            output.write(f"{self.primary_key}\n")
            for key in self.results["missing_in_df2"]:
                output.write(f"{key}\n")
            output.write("\n")

        if self.results["missing_in_df1"]:
            output.write("=== MISSING IN FILE 1 ===\n")
            output.write(f"{self.primary_key}\n")
            for key in self.results["missing_in_df1"]:
                output.write(f"{key}\n")
            output.write("\n")

        # Field mismatches
        if self.results["field_mismatches"]:
            output.write("=== FIELD MISMATCHES ===\n")
            output.write(f"{self.primary_key},Field,Value File 1,Value File 2\n")
            for mismatch in self.results["field_mismatches"]:
                key = mismatch["primary_key"]
                for field in mismatch["mismatched_fields"]:
                    output.write(f"{key},{field['field']},{field['value_file1']},{field['value_file2']}\n")

        return output.getvalue()


@app.post("/compare")
async def compare_csv_files(
        file1: UploadFile = File(...),
        file2: UploadFile = File(...),
        primary_key: str = Form(...)
):
    try:
        content1 = await file1.read()
        content2 = await file2.read()

        df1 = pd.read_csv(io.BytesIO(content1))
        df2 = pd.read_csv(io.BytesIO(content2))

        comparator = CSVComparator(df1, df2, primary_key)
        results = comparator.compare()

        return {
            "status": "success",
            "summary": {
                "total_rows_file1": len(df1),
                "total_rows_file2": len(df2),
                "missing_in_file2": len(results["missing_in_df2"]),
                "missing_in_file1": len(results["missing_in_df1"]),
                "rows_compared": results["total_compared"],
                "field_mismatches": len(results["field_mismatches"]),
                "quality_score": results["quality_score"]
            },
            "details": results
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/compare/export")
async def compare_and_export(
        file1: UploadFile = File(...),
        file2: UploadFile = File(...),
        primary_key: str = Form(...)
):
    try:
        content1 = await file1.read()
        content2 = await file2.read()

        df1 = pd.read_csv(io.BytesIO(content1))
        df2 = pd.read_csv(io.BytesIO(content2))

        comparator = CSVComparator(df1, df2, primary_key)
        comparator.compare()

        report_csv = comparator.generate_report_csv()

        return StreamingResponse(
            io.StringIO(report_csv),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CSV Comparison Tool</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #ffffff 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }

            .header h1 {
                font-size: 2em;
                margin-bottom: 10px;
            }

            .header p {
                opacity: 0.9;
            }

            .content {
                padding: 40px;
            }

            .upload-section {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 30px;
                margin-bottom: 20px;
            }

            .form-group {
                margin-bottom: 20px;
            }

            label {
                display: block;
                font-weight: 600;
                margin-bottom: 8px;
                color: #333;
            }

            input[type="file"] {
                width: 100%;
                padding: 12px;
                border: 2px dashed #667eea;
                border-radius: 8px;
                background: white;
                cursor: pointer;
                transition: all 0.3s;
            }

            input[type="file"]:hover {
                border-color: #764ba2;
                background: #f8f9fa;
            }

            input[type="text"] {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border 0.3s;
            }

            input[type="text"]:focus {
                outline: none;
                border-color: #667eea;
            }

            .btn {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s;
            }

            .btn:hover {
                transform: translateY(-2px);
            }

            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .results {
                margin-top: 30px;
                display: none;
            }

            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }

            .stat-card {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }

            .stat-card h3 {
                font-size: 2em;
                margin-bottom: 5px;
            }

            .stat-card p {
                opacity: 0.9;
                font-size: 0.9em;
            }

            .quality-score {
                background: white;
                border: 3px solid #667eea;
                color: #667eea;
                font-weight: bold;
            }

            .details-section {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }

            .details-section h3 {
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.3em;
            }

            .table-container {
                overflow-x: auto;
                max-height: 400px;
                overflow-y: auto;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                background: white;
            }

            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }

            th {
                background: #667eea;
                color: white;
                position: sticky;
                top: 0;
            }

            tr:hover {
                background: #f8f9fa;
            }

            .loading {
                display: none;
                text-align: center;
                padding: 30px;
            }

            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            
            .footer {
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                color: #666;
                border-top: 2px solid #e0e0e0;
            }

            .footer p {
                margin: 5px 0;
            }

            .footer .author {
                font-weight: 600;
                color: #1e3c72;
                font-size: 1.1em;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .error {
                background: #ff4444;
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                display: none;
            }

            .download-btn {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>CSV Comparison Tool</h1>
                <p>So sánh 2 file CSV và phân tích sự khác biệt</p>
            </div>

            <div class="content">
                <div class="upload-section">
                    <form id="uploadForm">
                        <div class="form-group">
                            <label>📁 File CSV 1:</label>
                            <input type="file" id="file1" accept=".csv" required>
                        </div>

                        <div class="form-group">
                            <label>📁 File CSV 2:</label>
                            <input type="file" id="file2" accept=".csv" required>
                        </div>

                        <div class="form-group">
                            <label>🔑 Primary Key (Column Name):</label>
                            <input type="text" id="primaryKey" value="fk_content_used_id" required>
                        </div>

                        <button type="submit" class="btn">🚀 So Sánh</button>
                    </form>
                </div>

                <div class="error" id="error"></div>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Đang xử lý dữ liệu...</p>
                </div>

                <div class="results" id="results">
                    <h2 style="margin-bottom: 20px; color: #333;">📈 Kết Quả So Sánh</h2>

                    <div class="summary-grid" id="summaryGrid"></div>

                    <div class="details-section" id="missingSection" style="display: none;">
                        <h3>❌ Dòng Thiếu</h3>
                        <div class="table-container" id="missingTable"></div>
                    </div>

                    <div class="details-section" id="mismatchSection" style="display: none;">
                        <h3>⚠️ Field Mismatches</h3>
                        <div class="table-container" id="mismatchTable"></div>
                    </div>

                    <button class="btn download-btn" id="downloadBtn">💾 Tải Báo Cáo CSV</button>
                </div>
            </div>
            
            <div class="footer">
                <p>© 2025 CSV Comparison Tool</p>
                <p class="author">Developed by Lê Gia Huy</p>
            </div>
        </div>

        <script>
            let currentResults = null;

            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                const file1 = document.getElementById('file1').files[0];
                const file2 = document.getElementById('file2').files[0];
                const primaryKey = document.getElementById('primaryKey').value;

                if (!file1 || !file2) {
                    showError('Vui lòng chọn đủ 2 file CSV');
                    return;
                }

                const formData = new FormData();
                formData.append('file1', file1);
                formData.append('file2', file2);
                formData.append('primary_key', primaryKey);

                showLoading(true);
                hideError();
                hideResults();

                try {
                    const response = await fetch('/compare', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Có lỗi xảy ra');
                    }

                    const data = await response.json();
                    currentResults = data;
                    displayResults(data);
                } catch (error) {
                    showError(error.message);
                } finally {
                    showLoading(false);
                }
            });

            function displayResults(data) {
                const summary = data.summary;
                const details = data.details;

                // Summary cards
                const summaryHTML = `
                    <div class="stat-card">
                        <h3>${summary.total_rows_file1}</h3>
                        <p>Tổng dòng File 1</p>
                    </div>
                    <div class="stat-card">
                        <h3>${summary.total_rows_file2}</h3>
                        <p>Tổng dòng File 2</p>
                    </div>
                    <div class="stat-card">
                        <h3>${summary.missing_in_file2 + summary.missing_in_file1}</h3>
                        <p>Dòng thiếu</p>
                    </div>
                    <div class="stat-card">
                        <h3>${summary.field_mismatches}</h3>
                        <p>Field khác biệt</p>
                    </div>
                    <div class="stat-card quality-score">
                        <h3>${summary.quality_score}%</h3>
                        <p>Chất lượng dữ liệu</p>
                    </div>
                `;
                document.getElementById('summaryGrid').innerHTML = summaryHTML;

                // Missing rows
                if (details.missing_in_df1.length > 0 || details.missing_in_df2.length > 0) {
                    let missingHTML = '<table><thead><tr><th>Primary Key</th><th>Thiếu ở</th></tr></thead><tbody>';

                    details.missing_in_df2.forEach(key => {
                        missingHTML += `<tr><td>${key}</td><td>File 2</td></tr>`;
                    });

                    details.missing_in_df1.forEach(key => {
                        missingHTML += `<tr><td>${key}</td><td>File 1</td></tr>`;
                    });

                    missingHTML += '</tbody></table>';
                    document.getElementById('missingTable').innerHTML = missingHTML;
                    document.getElementById('missingSection').style.display = 'block';
                }

                // Field mismatches
                if (details.field_mismatches.length > 0) {
                    let mismatchHTML = '<table><thead><tr><th>Primary Key</th><th>Field</th><th>Value File 1</th><th>Value File 2</th></tr></thead><tbody>';

                    details.field_mismatches.forEach(mismatch => {
                        mismatch.mismatched_fields.forEach(field => {
                            mismatchHTML += `
                                <tr>
                                    <td>${mismatch.primary_key}</td>
                                    <td>${field.field}</td>
                                    <td>${field.value_file1}</td>
                                    <td>${field.value_file2}</td>
                                </tr>
                            `;
                        });
                    });

                    mismatchHTML += '</tbody></table>';
                    document.getElementById('mismatchTable').innerHTML = mismatchHTML;
                    document.getElementById('mismatchSection').style.display = 'block';
                }

                document.getElementById('results').style.display = 'block';
            }

            document.getElementById('downloadBtn').addEventListener('click', async () => {
                const file1 = document.getElementById('file1').files[0];
                const file2 = document.getElementById('file2').files[0];
                const primaryKey = document.getElementById('primaryKey').value;

                const formData = new FormData();
                formData.append('file1', file1);
                formData.append('file2', file2);
                formData.append('primary_key', primaryKey);

                try {
                    const response = await fetch('/compare/export', {
                        method: 'POST',
                        body: formData
                    });

                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `comparison_report_${new Date().getTime()}.csv`;
                    a.click();
                } catch (error) {
                    showError('Không thể tải báo cáo');
                }
            });

            function showLoading(show) {
                document.getElementById('loading').style.display = show ? 'block' : 'none';
            }

            function hideResults() {
                document.getElementById('results').style.display = 'none';
            }

            function showError(message) {
                const errorDiv = document.getElementById('error');
                errorDiv.textContent = message;
                errorDiv.style.display = 'block';
            }

            function hideError() {
                document.getElementById('error').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)