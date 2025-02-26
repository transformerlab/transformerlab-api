## Usage

1. **Prepare Your Dataset:**
   - Ensure your dataset has clearly defined input and output columns
   - Verify that column names match the configured `input_col` and `output_col` parameters
   - Make sure your data is in a compatible format (CSV, JSON, etc.)

2. **Configure Evaluation Tasks:**
   - Define evaluation metrics using the `tasks` parameter:

     ```json
     {
       "name": "Contains Number",
       "expression": "\\d+",
       "return_type": "boolean"
     }
     ```

   - Create multiple tasks to evaluate different aspects of the output
   - Choose appropriate return types:
     - `boolean`: For yes/no evaluations
     - `number`: For scoring or counting matches

3. **Set Evaluation Parameters:**
   - Adjust the `limit` parameter to control the fraction of data to evaluate
   - Configure input and output column names to match your dataset

4. **Run the Evaluation:**
   - Execute the evaluation process
   - Monitor the progress as the plugin processes your dataset
   - Results will be stored and a detailed report will be generated.
