//document.querySelector('#generate-names').addEventListener('submit', loadNames);
//document.querySelector('#generate-models').addEventListener('onchange', loadModels);
//document.querySelector('#country').addEventListener('onchange', loadModels);
document.querySelector('#generate-names').addEventListener('submit', loadData);



var ml_type = "";


function change_modelSelection(e){
    console.log(e);
    var ml_type = document.getElementById("country").value;
    var model_value = ""

    if (ml_type == "Supervised"){
        model_value = "logisticRegression"
    }
    var ml_type = document.getElementById('genre').value;
    
}


function myFunction() {
    var x = document.getElementById("country").value;
    var selectBox = document.getElementById("genre");
    console.log("x", x)
    if (x === "supervised"){
        selectBox.innerHTML = '<option value="">-- Select --</option><option value="logisticRegression">LogisticRegression</option><option value="linearRegression">LinearRegression</option>';
    } else {
        selectBox.innerHTML = '<option value="clustering">Clustering</option>'
   }
}





// Execute the function to query the API
function loadNames(e) {
     e.preventDefault();


     // Read the values from the form and create the variables
     const ml_type = document.getElementById('country').value;

     console.log("ml_type", ml_type)
    
    const genre = document.getElementById('genre').value;
    console.log("genre", genre)


     // Build the URL
     let url = 'http://localhost:5000/displayHeaders?';
     //let url = "http://localhost:5500/"
     
     console.log("url", url)

     let fileName = "Iris.csv";
     
     // Browse the file  and append to the url
     if (fileName != ""){
         url += "fileName=Iris.csv&";
     
    
     
     // Ajax Call
     const xhr = new XMLHttpRequest();

     // Open the connection
     xhr.open('GET', url, true );

     // Execute the function
     xhr.onload = function() {
          if(this.status === 200) {
               const names = JSON.parse( this.responseText );

               console.log("names", names)
               
            //    // Insert into the HTML

               let html = '<h2>Generated Names</h2>';
               html += '<ul class="list">';
            //    names.forEach(function(name){
            //         html += `
            //              <li>${name.name}</li>
            //         `;
            //    });
            names.forEach(function(name){
                html += `
                     <li>${name}</li>
                `;
           });
               html += '</ul>';

               document.querySelector('#result').innerHTML = html;
          }
     }

     // Send the request
     xhr.send(); }
     else {
         console.log("file name not present");
     }
}




function loadData(e) {

    var formData = new FormData();

    console.log("generating file");

    console.log(document.getElementById('inputFile'));

    var fileName = document.getElementById('inputFile').files[0];



    console.log(fileName)

    var file_name_csv = fileName.name

    console.log("fileName", file_name_csv)

    // Build the URL
    let url = 'http://localhost:5000/postData?';


    // Ajax Call
     const xhr = new XMLHttpRequest();

    xhr.open('post', url, true);

    //formData.append("fileName", file);
    formData.append("fileName", fileName);
    //formData.append("fileName", fileName);


    xhr.send(formData);


     // Execute the function
     xhr.onload = function() {
        if(this.status === 200) {
             const names = JSON.parse( this.responseText );

             console.log("names", names)
             
          //    // Insert into the HTML

             let html = '<h3>Input Data</h3>';
        //html += '<ul class="list">';
          //    names.forEach(function(name){
          //         html += `
          //              <li>${name.name}</li>
          //         `;
          //    });
        //   names.forEach(function(name){
        //       html += `
        //            <li>${name}</li>
        //       `;
        //  });
        //      html += '</ul>';

             html += names.data


             document.querySelector('#result').innerHTML = html;


            var list_names = names.headers_list

            var list_nameHeader = "<h3>Input Variables</h3>"
            list_nameHeader =  '<ol class="list">'
            list_names.forEach(function(name){
                list_nameHeader += `
                             <li>${name}</li>
                        `;
                   });
                   list_nameHeader += '</ol>';


             document.querySelector('#selecting_features').innerHTML = list_nameHeader;
        }
   }







    

}