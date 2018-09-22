class ML_ClickAPI{
    // Get recipe by name
    async displayData(name) {
         // Search by name
         const apiResponse = await fetch(`${name}`);
         // Returns a json respone
         const data = await apiResponse.json();

         return {
              data
         }
    }