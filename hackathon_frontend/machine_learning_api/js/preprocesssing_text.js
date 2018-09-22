document.querySelector('#training-model').addEventListener('submit', trainingModel);
document.querySelector('#evaluation-model').addEventListener('submit', evaluationModel);
document.querySelector('#generate-names').addEventListener('submit', loadData);



function myFunction() {
    var x = document.getElementById("modelType").value;
    var selectBox = document.getElementById("model_selection");
    console.log("x", x)
    if (x === "supervised"){
        selectBox.innerHTML = '<option value="">-- Select --</option><option value="logisticRegression">LogisticRegression</option><option value="linearRegression">LinearRegression</option>'
        + '<option value="randomForestClassifier">RandomForestClassifier</option><option value="supportVectorMachines">Support Vector Machines</option><option value="neuralNetworks">Neural Networks(MLP Classifier)</option>';
    } else if (x === "unsupervised"){
        selectBox.innerHTML = '<option value="">-- Select --</option><option value="kMeansClustering">K-Means Clustering</option><option value="kMeansClustering">Hierarchical Clustering</option>'
   } else {
    selectBox.innerHTML = '<option value="">-- Select --</option><option value="image_Recognition">Image Recognition</option>'
   }
}

var formData = new FormData();



function loadData(e) {


    console.log("generating file");

    console.log(document.getElementById('inputFile'));

    var fileName = document.getElementById('inputFile').files[0];

    console.log(fileName);

    const model_type = document.getElementById("modelType").value;


    const model_selection = document.getElementById("model_selection").value;

    
    console.log("model_selection", model_selection);

    // Store it in the local storage
    localStorage.setItem('selected_model', model_selection);


    var file_name_csv = fileName.name;

    console.log("fileName", file_name_csv);

    // Build the URL
    let url = ""

    if (model_type != "imageRecognition"){

        url = 'http://localhost:5000/postData?';

    } else {

            url = 'http://localhost:5000/detect_labels?';
    }


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
        
        if  (model_type != "imageRecognition"){

             const names = JSON.parse(this.responseText );

             console.log("names", names)
             
          //    // Insert into the HTML
        console.log("model_type", model_type)

             let html = '<h3>Input Data</h3>';

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


             //document.querySelector('#selecting_features').innerHTML = list_nameHeader;
        } else {
        
        const names =  this.responseText;
        console.log("names_description",  names)
        let html = "The image is of <b>" + names + "</b>"
          alert(html)
          document.querySelector('#result').innerHTML = html;
    }  }
   }
}



  // Execute the function to Train the model
  function trainingModel(e) {
    e.preventDefault();


    console.log("formData", formData)

    // Get it from the local storage
     var selected_model_train = localStorage.getItem('selected_model');

     console.log("selected_model_train", selected_model_train)


    // Read the values from the form and create the variables
    const cleaning_operations_type = document.getElementById('preprocessing_text').value;

    console.log("cleaning_operations", cleaning_operations_type);
   
   const vectorizer = document.getElementById('vectorizer_selection').value;
   console.log("vectorizer", vectorizer)


    // Build the URL
    let url = ""

    if (selected_model_train == "logisticRegression")  {

    url = 'http://localhost:5000/model_fit';

    } else {
        url = 'http://localhost:5000/model_fit';
    }
    
    console.log("url", url)

    
    // Browse the file  and append to the url
    
   
    
    // Ajax Call
    const xhr = new XMLHttpRequest();

    // Open the connection
    xhr.open('GET', url, true );

    // Execute the function
    xhr.onload = function() {
         if(this.status === 200) {
             console.log("this.responseText ", this.responseText );
              const value_response = JSON.parse( this.responseText );
              //const value_response = this.responseText;


              alert(value_response);

              let html = value_response;


              document.querySelector('#result_training').innerHTML = html;
         }
    }

    // Send the request
    xhr.send();

}





// Execute the function to Train the model
function evaluationModel(e) {
    e.preventDefault();


    // Build the URL
    let url = 'http://localhost:5000/model_evaluate';
    
    console.log("url", url)

    
    // Browse the file  and append to the url
    
   
    
    // Ajax Call
    const xhr = new XMLHttpRequest();

    // Open the connection
    xhr.open('GET', url, true );

    // Execute the function
    xhr.onload = function() {
         if(this.status === 200) {
              const value_response = JSON.parse( this.responseText );
              //const value_response = this.responseText;

              console.log(value_response)

              const accuracy_result = value_response.prediction_result;


              alert(accuracy_result);

              let html = "<p>" + accuracy_result + "</p>";

              html += value_response.classfication_report;


              document.querySelector('#result_evaluation').innerHTML = html;
         }
    }

    // Send the request
    xhr.send();

}