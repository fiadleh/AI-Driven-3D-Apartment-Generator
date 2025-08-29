var network = null;
var seed = 2;

// label to color map
var room_to_color = {"living": '#EE4D4D', "kitchen": '#C67C7B', "bedroom": '#FFD274',
                     "bathroom": '#BEBEBE', "balcony": '#BFE3E8', "entrance": '#7BA779',
                     "dining": '#E87A90', "study": '#FF8C69', "storage": '#1F849B'};

// Template - 1-Bedroom
var template_nodes_1 = [
  { id: 0, label: "bedroom", color: '#FFD274'},
  { id: 1, label: "bathroom", color: '#BEBEBE'},
  { id: 2, label: "living", color: '#EE4D4D'},
  { id: 'outside', label: "outside", color: '#727171'},
  { id: 4, label: "balcony", color: '#BFE3E8'},
];

var template_edges_1 = [
  { from: 0, to: 1 , color: '#D3A2C7', width: 3},
  { from: 2, to: 0 , color: '#D3A2C7', width: 3},
  { from: 2, to: 1 , color: '#D3A2C7', width: 3},
  { from: 2, to: 'outside' , color: '#D3A2C7', width: 3},
  { from: 2, to: 4 , color: '#D3A2C7', width: 3},
  { from: 0, to: 4 , color: '#D3A2C7', width: 3},
];

// Template - Two-bedroom suite
var template_nodes_2 = [
  { id: 0, label: "bedroom", color: '#FFD274'},
  { id: 1, label: "bedroom", color: '#FFD274'},
  { id: 2, label: "bathroom", color: '#BEBEBE'},
  { id: 3, label: "bathroom", color: '#BEBEBE'},
  { id: 4, label: "balcony", color: '#BFE3E8'},
  { id: 5, label: "living", color: '#EE4D4D'},
  { id: 'outside', label: "outside", color: '#727171'},
];

var template_edges_2 = [
  { from: 5, to: 1 , color: '#D3A2C7', width: 3},
  { from: 5, to: 0 , color: '#D3A2C7', width: 3},
  { from: 5, to: 3 , color: '#D3A2C7', width: 3},
  { from: 5, to: 4 , color: '#D3A2C7', width: 3},
  { from: 'outside', to: 5 , color: '#D3A2C7', width: 3},
  { from: 0, to: 2 , color: '#D3A2C7', width: 3},
  { from: 1, to: 3 , color: '#D3A2C7', width: 3},
];


// Template - Three-bedroom suite
var template_nodes_3 = [
  { id: 0, label: "bedroom", color: '#FFD274'},
  { id: 1, label: "bedroom", color: '#FFD274'},
  { id: 2, label: "bedroom", color: '#FFD274'},
  { id: 3, label: "bathroom", color: '#BEBEBE'},
  { id: 4, label: "bathroom", color: '#BEBEBE'},
  { id: 5, label: "kitchen", color: '#C67C7B'},
  { id: 6, label: "living", color: '#EE4D4D'},
  { id: 'outside', label: "outside", color: '#727171'},
  { id: 8, label: "balcony", color: '#BFE3E8'},
];

var template_edges_3 = [
  { from: 4, to: 2 , color: '#D3A2C7', width: 3},
  { from: 6, to: 0 , color: '#D3A2C7', width: 3},
  { from: 6, to: 1 , color: '#D3A2C7', width: 3},
  { from: 6, to: 2 , color: '#D3A2C7', width: 3},
  { from: 6, to: 3 , color: '#D3A2C7', width: 3},
  { from: 6, to: 5 , color: '#D3A2C7', width: 3},
  { from: 'outside', to: 6 , color: '#D3A2C7', width: 3},
  { from: 8, to: 2 , color: '#D3A2C7', width: 3},
];

// Template - Three-bedroom suite
var template_nodes_4 = [
  { id: 0, label: "bedroom", color: '#FFD274'},
  { id: 1, label: "bedroom", color: '#FFD274'},
  { id: 2, label: "bedroom", color: '#FFD274'},
  { id: 3, label: "bathroom", color: '#BEBEBE'},
  { id: 4, label: "dining", color: '#E87A90'},
  { id: 5, label: "kitchen", color: '#C67C7B'},
  { id: 6, label: "living", color: '#EE4D4D'},
  { id: 'outside', label: "outside", color: '#727171'},
  { id: 8, label: "balcony", color: '#BFE3E8'},
];

var template_edges_4 = [
  { from: 4, to: 2 , color: '#D3A2C7', width: 3},
  { from: 6, to: 0 , color: '#D3A2C7', width: 3},
  { from: 6, to: 1 , color: '#D3A2C7', width: 3},
  { from: 6, to: 2 , color: '#D3A2C7', width: 3},
  { from: 6, to: 3 , color: '#D3A2C7', width: 3},
  { from: 6, to: 5 , color: '#D3A2C7', width: 3},
  { from: 'outside', to: 6 , color: '#D3A2C7', width: 3},
  { from: 8, to: 2 , color: '#D3A2C7', width: 3},
];
// create a network
var container = document.getElementById("mynetwork");


// legend
function add_legend(data) {
  var mynetwork = document.getElementById("mynetwork");
  var x = -mynetwork.clientWidth / 2 + 50;
  var y = -mynetwork.clientHeight / 2 - 50;
  var step = 60;
  var i = 0;
  for (var key in room_to_color) {
    // console.log("legend_" + key);
      data.nodes.add({
              x: x + i*step,
              y: y,
              id: "legend_" + key,
              label: key,
              color: room_to_color[key],
              group: key,
              shape: "square",
              value: 1,
              fixed: true,
              physics: false,
            });
      i++;
    }
  }

function destroy() {
  if (network !== null) {
    network.destroy();
    network = null;
  }
}

function draw(data) {
  //print(' -------- Draw --------------')
  //alert("-------- Draw --------------");
  destroy();

  // create a network
  var container = document.getElementById("mynetwork");
  var options = {
    nodes: { borderWidth: 2 },
    layout: { randomSeed: seed },
    locales: {
      en: {
        edit: 'Edit',
        del: 'Delete selected',
        back: 'Back',
        addNode: 'Add Node',
        addEdge: 'Add Edge',
        editNode: 'Edit Node',
        editEdge: 'Edit Edge',
        addDescription: 'Select a room type from the legend.',
        edgeDescription: 'Click on a node and drag the edge to another node to connect them.',
        editEdgeDescription: 'Click on the control points and drag them to a node to connect to it.',
        createEdgeError: 'Cannot link edges to a cluster.',
        deleteClusterError: 'Clusters cannot be deleted.',
        editClusterError: 'Clusters cannot be edited.'
      }
    },
    manipulation: {
      addNode: function (nodeData, callback) {
        // filling in the popup DOM elements
        if (network.getSelectedNodes().length > 0){
          selectedNode = network.getSelectedNodes()[0]
          var nodesLength = data.nodes.getIds().length;
          if (String(selectedNode).includes('legend') && nodesLength <= 22){
            editNode(nodeData, selectedNode, clearNodePopUp, callback);
          }
          else if(nodesLength > 22){
            alert("Err: trying to add too many nodes.");
            callback(null);
          }
          else {
            alert("Err: node must be selected from the legend.");
            callback(null);
          }
        }
      },
      deleteNode: function(data, callback) {
        if (String(data.nodes[0]).includes('legend_') == true || String(data.nodes[0]).includes('outside') == true){
          alert("Err: this node can not be deleted.");
          callback(null);
        }
        else{
          callback(data);
          //generate(); 
        }
      },
      addEdge: function (data, callback) {
        if (data.from == data.to) {
          alert("Err: trying to add self connections.");
          callback(null);
          return;
        }
        document.getElementById("edge-operation").innerText = "Add Edge";
        saveEdgeData(data, callback);
      },
      deleteEdge: function(data, callback) {
        callback(data);
        //generate();
      },
    },
  };

  //update_prompts(data.nodes)
  network = new vis.Network(container, data, options);
  network.fit()
}

function editNode(data, label, cancelAction, callback) {
  var mynetwork = document.getElementById("mynetwork");
  var x = -mynetwork.clientWidth / 2 + 200;
  var y = -mynetwork.clientHeight / 2 + 200;
  data.x = x;
  data.y = y;
  data.label = label.split("_")[1];
  data.color = room_to_color[data.label]
  clearNodePopUp();
  callback(data);
  //generate();
}

// Callback passed as parameter is ignored
function clearNodePopUp() {
  document.getElementById("node-saveButton").onclick = null;
  document.getElementById("node-cancelButton").onclick = null;
  document.getElementById("node-popUp").style.display = "none";
}

function cancelNodeEdit(callback) {
  clearNodePopUp();
  callback(null);
}

function saveEdgeData(data, callback) {
  if (typeof data.to === "object") data.to = data.to.id;
  if (typeof data.from === "object") data.from = data.from.id;
  data.color = '#D3A2C7';
  data.width = 3;
  callback(data);
  //generate();
}

function checker_reset(k){
  $(".check"+k.toString()).attr("class", "check"+k.toString());
  $(".fill"+k.toString()).attr("class", "fill"+k.toString());
  $(".path"+k.toString()).attr("class", "path"+k.toString());
}

function checker_complete(k){
  $(".check"+k.toString()).attr("class", "check"+k.toString()+" check-complete"+k.toString()+" success");
  $(".fill"+k.toString()).attr("class", "fill"+k.toString()+" fill-complete"+k.toString()+" success");
  $(".path"+k.toString()).attr("class", "path"+k.toString()+" path-complete"+k.toString());
}

function update_prompts(nodes, types) {
  // Reset the prompts container
  var prompts_container = document.getElementById("text_prompts");
  prompts_container.innerHTML = '';
  // Define the text arrays for each room type
  const bedroomTexts = ["Add a corner side table with a round top to the left of a black and silver pendant lamp with lights", 
    "Put a grey double bed with headboard and pillows to the left of a nightstand", 
    "Position a wardrobe right of a pendant lamp. And add a black and brown double bed with a cover next to a black rattan pendant lamp.", 
    'Put a gray wardrobe with doors and shelves right of a black and grey tv stand with drawers. Next, arrange a gray wardrobe with doors and shelves right of a grey single bed with a pillow.',
    'Set up a black wardrobe with doors and drawers to the left of a plaid single bed with pillows.', 
    'Position a gray double bed with brown pillows below a wooden pendant lamp with a wooden shade.',
    'Arrange a wooden nightstand with a hole to the left of a wooden wardrobe with doors and drawers.'];
  
    const livingTexts = ["Put a black and silver lounge chair to the right of a dining table.", 
      "Place a blue multi-seat sofa with pillows behind a corner side table.", 
      "Arrange a dining chair behind a black and brown dining table with black legs.",
      'Position a black circular dining table right of a classic chinese chair with a cushion.'];

  const diningTexts = ["Set up a brass pendant lamp with lights above a dining table with a marble top", 
    "Place a black pendant lamp with hanging balls above a grey dining table with round top. Next, position a grey dining chair to the close right below of a black pendant lamp with hanging balls", 
    "Place an L-shaped sofa behind a grey marble desk. Then, position a cabinet with shelves in front of a lounge chair",
    'Arrange a dining table below a black pendant lamp with a handle.',
    'Arrange a dining table below a black pendant lamp with a handle.'
  ];

  // Function to get random text for the given room type
  function getRandomText(roomType) {
      let texts;
      switch(roomType) {
          case "bedroom":
              texts = bedroomTexts;
              break;
          case "living":
              texts = livingTexts;
              break;
          case "dining":
              texts = diningTexts;
              break;
          default:
              texts = ["Default description"];
      }
      // Return a random element from the selected text array
      return texts[Math.floor(Math.random() * texts.length)];
  }
  
  // Update prompts container with text areas for each room
  nodes.forEach(nodeId => {
    var roomType = types[nodeId];
    if (roomType === "bedroom" || roomType === "living" || roomType === "dining") {
      var textArea = document.createElement("textarea");
      textArea.id = "prompt_" + roomType + "_"+ nodeId;
      textArea.className = "form-control"
      textArea.placeholder = `Enter description for ${roomType} ${nodeId}`;
      // Set the random text as the value of the textarea
      textArea.value = getRandomText(roomType);
      prompts_container.appendChild(textArea);
    }
  });
  var generateButton = document.createElement("button");
  generateButton.id = "generate_3d";
  generateButton.className = "generateButton";
  generateButton.innerText = "Generate 3D";
  prompts_container.appendChild(generateButton);

  // Set up button click event to make an xhr POST request
  generateButton.addEventListener("click", function() {
    // Collect text from all textareas
    var textAreas = prompts_container.querySelectorAll("textarea");
    var promptsData = {};
    
    textAreas.forEach(textArea => {
      promptsData[textArea.id] = textArea.value;
    });

    var xhr = new XMLHttpRequest();
    xhr.open("POST", 'http://localhost:5000/generate3d', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onreadystatechange = function() {
      if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
        console.log("3D Generation Request Successful");
        console.log("Response:", xhr.responseText);
      } else if (xhr.readyState === XMLHttpRequest.DONE) {
        console.log("3D Generation Request Failed");
        console.log("Response:", xhr.responseText);
      }
    };

    xhr.send(JSON.stringify(promptsData));
  });

  //var adjustButton = document.createElement("button");
  //adjustButton.id = "adjust";
  //adjustButton.className = "generateButton";
  //adjustButton.innerText = "Adjust";
  //prompts_container.appendChild(adjustButton);
  // Set up button click event to make an xhr POST request
  /*
  adjustButton.addEventListener("click", function() {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", 'http://localhost:5000/adjust', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onreadystatechange = function() {
      if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
        console.log("3D Generation Request Successful");
        console.log("Response:", xhr.responseText);
      } else if (xhr.readyState === XMLHttpRequest.DONE) {
        console.log("3D Generation Request Failed");
        console.log("Response:", xhr.responseText);
      }
    };

    xhr.send();
  });
  */
  /******************************************************************************************************** */
}

function generate() {
  console.log("Generating ...");

  // Start checker
  for (var i = 0; i < 6; i++){
    checker_reset(i);
  }

  // Get current graph
  var nodeIndices = data.nodes.getIds();
  var edgesIndices = data.edges.getIds();
  var nodes = [];
  var types = {};
  var edges = [];
  var edgesObj = [];

  // Extract nodes and their types
  nodeIndices.forEach((nodeId, index) => {
    if (!nodeId.toString().includes("legend_")) {
      nodes.push(nodeId.toString());
      types[nodeId.toString()] = data.nodes.get(nodeId).label;
    }
  });

  // Extract edges
  edgesIndices.forEach(edgeId => {
    edgesObj.push(data.edges.get(edgeId));
  });

  nodes.forEach((nodeId1, i) => {
    nodes.forEach((nodeId2, j) => {
      if (j < i) {
        edgesObj.forEach(edge => {
          if ((edge.from == nodeId1 && edge.to == nodeId2) || (edge.from == nodeId2 && edge.to == nodeId1)) {
            edges.push([i, j]);
          }
        });
      }
    });
  });

  let graph_info = { "nodes": types, "edges": edges };
  console.log("Graph Info: ", graph_info);

  // Reset the prompts container
  let prompts_container = document.getElementById("text_prompts");
  prompts_container.innerHTML = '';

  // Define the text arrays for each room type
  //const bedroomTexts = ["Add a corner side table with a round top to the left of a black and silver pendant lamp with lights", "Spacious room with a queen-sized bed", "Position a wardrobe right of a pendant lamp. And add a black and brown double bed with a cover next to a black rattan pendant lamp."];
  //const livingTexts = ["Let the room be in gray style", "Comfortable seating with a large TV", "Modern living room with stylish decor"];
  //const diningTexts = ["Set up a brass pendant lamp with lights above a dining table with a marble top", "Place a black pendant lamp with hanging balls above a grey dining table with round top. Next, position a grey dining chair to the close right below of a black pendant lamp with hanging balls", "Place an L-shaped sofa behind a grey marble desk. Then, position a cabinet with shelves in front of a lounge chair"];

  const bedroomTexts = ["Add a corner side table with a round top to the left of a black and silver pendant lamp with lights", 
    "Put a grey double bed with headboard and pillows to the left of a nightstand", 
    "Position a wardrobe right of a pendant lamp. And add a black and brown double bed with a cover next to a black rattan pendant lamp.", 
    'Place a black and grey double bed in front of a wardrobe. Then, arrange a black and white square nightstand in front of a black wardrobe with hanging clothes.',
    'Put a gray wardrobe with doors and shelves right of a black and grey tv stand with drawers. Next, arrange a gray wardrobe with doors and shelves right of a grey single bed with a pillow.',
    'Set up a black wardrobe with doors and drawers to the left of a plaid single bed with pillows.', 
    'Position a gray double bed with brown pillows below a wooden pendant lamp with a wooden shade.',
    'Arrange a wooden nightstand with a hole to the left of a wooden wardrobe with doors and drawers.'];
  
    const livingTexts = ["Put a black and silver lounge chair to the right of a dining table.", 
      "Place a blue multi-seat sofa with pillows behind a corner side table.", 
      "Position a white circular dining table right of a classic chair with a cushion.",
      "Place a blue multi-seat sofa with pillows behind a corner side table. Next, place a black glass dining table with a metal base left of a coffee table.",
      'Position a black circular dining table right of a classic chinese chair with a cushion.'];
      
  const diningTexts = ["Set up a brass pendant lamp with lights above a dining table with a marble top", 
    "Place a black pendant lamp with hanging balls above a grey dining table with round top. Next, position a grey dining chair to the close right below of a black pendant lamp with hanging balls", 
    "Place an L-shaped sofa behind a grey marble desk. Then, position a cabinet with shelves in front of a lounge chair",
    'Arrange a dining table below a black pendant lamp with a handle.',
    'Arrange a dining table below a black pendant lamp with a handle.'
  ];
  // Function to get random text for the given room type
  function getRandomText(roomType) {
    let texts;
    switch(roomType) {
      case "bedroom":
        texts = bedroomTexts;
        break;
      case "living":
        texts = livingTexts;
        break;
      case "dining":
        texts = diningTexts;
        break;
      default:
        texts = ["Default description"];
    }
    return texts[Math.floor(Math.random() * texts.length)];
  }

  // Update prompts container with text areas for each room
  nodes.forEach(nodeId => {
    let roomType = types[nodeId];
    if (roomType === "bedroom" || roomType === "living" || roomType === "dining") {
      let textArea = document.createElement("textarea");
      textArea.id = `prompt_${roomType}_${nodeId}`;
      textArea.className = "form-control";
      textArea.placeholder = `Enter description for ${roomType} ${nodeId}`;
      textArea.value = getRandomText(roomType);
      prompts_container.appendChild(textArea);
    }
  });

  // Create container to hold the inputs
  const inputContainer = document.createElement('input-container');

  // Default values, hints, and labels for the inputs
  const inputsData = [
      { value: 2.8, hint: 'Wall Height', label: 'Wall Height: ', id: 'wall_height' },
      { value: 0.1, hint: 'Wall Thickness', label: 'Wall Thickness: ', id: 'wall_thickness' },
      { value: 2.01, hint: 'Door Height', label: 'Door Height: ', id: 'door_height' },
      { value: 0.86, hint: 'Door Width', label: 'Door Width: ', id: 'door_width' },
      { value: 1.23, hint: 'Window Width', label: 'Window Width: ', id: 'window_width' },
      { value: 1.48, hint: 'Window Height', label: 'Window Height: ', id: 'window_height' },
      { value: 1.0, hint: 'Window Height from floor', label: 'Window Height from floor: ', id: 'window_height_from_floor' },
  ];

  // Create and append the labeled inputs
  inputsData.forEach(data => {
      const labeledInputElement = createLabeledNumericalInput(data.value, data.hint, data.label, data.id);
      inputContainer.appendChild(labeledInputElement);
      inputContainer.appendChild(document.createElement('br')); // Line break for better layout
  });

  prompts_container.appendChild(inputContainer);

  let generateButton = document.createElement("button");
  generateButton.id = "generate_3d";
  generateButton.className = "generateButton";
  generateButton.innerText = "Generate 3D";
  prompts_container.appendChild(generateButton);

  // Set up button click event to make an xhr POST request
  generateButton.addEventListener("click", function() {
    let textAreas = prompts_container.querySelectorAll("textarea");
    let combinedData = {};
    let promptsData = {};

    textAreas.forEach(textArea => {
      promptsData[textArea.id] = textArea.value;
    });
    combinedData['text_prompts'] = promptsData;
    combinedData['wall_height'] = document.getElementById('wall_height').value;
    combinedData['wall_thickness'] = document.getElementById('wall_thickness').value;
    combinedData['door_height'] = document.getElementById('door_height').value;
    combinedData['door_width'] = document.getElementById('door_width').value;
    combinedData['window_width'] = document.getElementById('window_width').value;
    combinedData['window_height'] = document.getElementById('window_height').value;
    combinedData['window_height_from_floor'] = document.getElementById('window_height_from_floor').value;
    let xhr = new XMLHttpRequest();
    xhr.open("POST", 'http://localhost:5000/generate3d', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onreadystatechange = function() {
      if (xhr.readyState === XMLHttpRequest.DONE) {
        if (xhr.status === 200) {
          console.log("3D Generation Request Successful");
          console.log("Response:", xhr.responseText);
        } else {
          console.log("3D Generation Request Failed");
          console.log("Response:", xhr.responseText);
        }
      }
    };

    xhr.send(JSON.stringify(combinedData));
  });

  //let adjustButton = document.createElement("button");
  //adjustButton.id = "adjust";
  //adjustButton.className = "generateButton";
  //adjustButton.innerText = "Adjust";
  //prompts_container.appendChild(adjustButton);
  /*
  adjustButton.addEventListener("click", function() {
    let xhr = new XMLHttpRequest();
    xhr.open("POST", 'http://localhost:5000/adjust', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onreadystatechange = function() {
      if (xhr.readyState === XMLHttpRequest.DONE) {
        if (xhr.status === 200) {
          console.log("Adjustment Request Successful");
          console.log("Response:", xhr.responseText);
        } else {
          console.log("Adjustment Request Failed");
          console.log("Response:", xhr.responseText);
        }
      }
    };

    xhr.send();
  });
  */

  // Function to create a numerical input element with a hint and label
  function createLabeledNumericalInput(defaultValue, hint, labelText, id) {
      // Create a div to hold the label and input together
      const container = document.createElement('div');
      
      // Create the label
      const label = document.createElement('label');
      label.textContent = labelText;
      container.appendChild(label);

      // Create the input element
      const input = document.createElement('input');
      input.id = id;
      input.type = 'number';
      input.value = defaultValue;
      input.step = 'any'; // Allow floating-point numbers
      input.placeholder = hint; // Add hint as a placeholder
      container.appendChild(input);

      return container;
  }

  
/*
  const container = document.createElement('div');
  let wall_height_input = document.createElement('input');
  wall_height_input.id = "adjust";
  wall_height_input.type = 'number';
  wall_height_input.step = 'any';
  wall_height_input.value = 2.8;
  wall_height_input.placeholder = "wall height";
  prompts_container.appendChild(wall_height_input);
*/
  // Generate layout
  let xhr = new XMLHttpRequest();
  let _ptr = 0;
  let num_iters = 1;
  let n_samples = 6;
  let _tracker = 0;
  xhr.onreadystatechange = function() {
    if (this.status === 200) {
      let data_stream = this.responseText.split('<stop>');
      for (let i = _tracker; i < data_stream.length; i++) {
        let image_data = data_stream[i];
        if (Math.floor(_ptr / num_iters) == 0 && image_data !== "" && _ptr < n_samples * num_iters) {
          let svg_element = document.getElementById("lgContainer");
          lgContainer.innerHTML = image_data;
          let svg = lgContainer.firstChild;
          scaleSVG(svg, 2.2);
          let progress = document.getElementById("progress");
          progress.innerHTML = "Sample Generated House Layout";
          _ptr++;
          if (Math.floor(_ptr / num_iters) > 0) {
            checker_complete(0);
          }
        } else if (image_data !== "" && _ptr < n_samples * num_iters) {
          let smContainer = document.getElementById("sm_img_" + (Math.floor(_ptr / num_iters) - 1));
          smContainer.innerHTML = image_data;
          smContainer.onclick = function() {
            let miniDisplay = document.getElementById($(this).attr('id'));
            let miniDisplaySVG = miniDisplay.firstChild;
            let largeDisplay = document.getElementById("lgContainer");
            let largeDisplaySVG = largeDisplay.firstChild;
            let s = new XMLSerializer();
            miniDisplay.innerHTML = s.serializeToString(largeDisplaySVG);
            scaleSVG(miniDisplay.firstChild, 1.0 / 2.2);
            scaleSVG(miniDisplay.firstChild, 0.49);
            largeDisplay.innerHTML = s.serializeToString(miniDisplaySVG);
            scaleSVG(largeDisplay.firstChild, 1.0 / 0.49);
            scaleSVG(largeDisplay.firstChild, 2.2);
          };
          let svg = smContainer.firstChild;
          scaleSVG(svg, 0.49);
          checker_complete(Math.floor(_ptr / num_iters) - 1);
          _ptr++;
        }
      }
      _tracker += (data_stream.length - _tracker);
      if (_tracker == data_stream.length) {
        checker_complete(Math.floor(_ptr / num_iters) - 1);
      }
    }
  };
  console.log(graph_info);
  xhr.open("POST", 'http://localhost:5000/generate', true);
  xhr.setRequestHeader('Content-Type', 'text/plain');
  xhr.send(JSON.stringify(graph_info));
}

function generate222() {
  
  //alert(' -- Gen ---')
  // start checker
  for (var i = 0; i < 6; i++){
    checker_reset(i);
  }
  // get current graph
  var nodeIndices = data.nodes.getIds();
  var edgesIndices = data.edges.getIds();
  var nodes = new Array();
  var types = new Object();
  var edges = new Array();
  var edgesObj = new Array();
  for (var i = 0; i < nodeIndices.length; i++) {
      if (nodeIndices[i].toString().includes("legend_") == false){
        nodes.push(nodeIndices[i].toString());
        types[i.toString()] = data.nodes.get(nodeIndices[i]).label;
      }
  }
  for (var i = 0; i < edgesIndices.length; i++) {
    edgesObj.push(data.edges.get(edgesIndices[i]));
  }
  for (var i = 0; i < nodes.length; i++) {
    for (var j = 0; j < nodes.length; j++) {
      if (j < i){
        for (var k = 0; k < edgesObj.length; k++) {
          if ((edgesObj[k].from == nodes[i] && edgesObj[k].to == nodes[j])||(edgesObj[k].from == nodes[j] && edgesObj[k].to == nodes[i])){
            edges.push([i, j]);
          }
        }
      }
    }
  }
  graph_info = ({"nodes":types, "edges":edges});
  

  /******************************************************************************************************** */
  // Reset the prompts container
  //alert(' -- Reset the prompts container ---')
  //alert(graph_info)
  var prompts_container = document.getElementById("text_prompts");
  prompts_container.innerHTML = '';
  // Define the text arrays for each room type
  const bedroomTexts = ["Add a corner side table with a round top to the left of a black and silver pendant lamp with lights", "Spacious room with a queen-sized bed", "Position a wardrobe right of a pendant lamp. And add a black and brown double bed with a cover next to a black rattan pendant lamp."];
  const livingTexts = ["Let the room be in gray style", "Comfortable seating with a large TV", "Modern living room with stylish decor"];
  const diningTexts = ["Set up a brass pendant lamp with lights above a dining table with a marble top", "Place a black pendant lamp with hanging balls above a grey dining table with round top. Next, position a grey dining chair to the close right below of a black pendant lamp with hanging balls", "Place an L-shaped sofa behind a grey marble desk. Then, position a cabinet with shelves in front of a lounge chair"];

  // Function to get random text for the given room type
  function getRandomText(roomType) {
      let texts;
      switch(roomType) {
          case "bedroom":
              texts = bedroomTexts;
              break;
          case "living":
              texts = livingTexts;
              break;
          case "dining":
              texts = diningTexts;
              break;
          default:
              texts = ["Default description"];
      }
      // Return a random element from the selected text array
      return texts[Math.floor(Math.random() * texts.length)];
  }
  
  // Update prompts container with text areas for each room
  nodes.forEach(nodeId => {
    var roomType = types[nodeId];
    if (roomType === "bedroom" || roomType === "living" || roomType === "dining") {
      var textArea = document.createElement("textarea");
      textArea.id = `prompt_${roomType}_${nodeId}`;
      textArea.className = "form-control"
      textArea.placeholder = `Enter description for ${roomType} ${nodeId}`;
      // Set the random text as the value of the textarea
      textArea.value = getRandomText(roomType);
      prompts_container.appendChild(textArea);
    }
  });
  var generateButton = document.createElement("button");
  generateButton.id = "generate_3d";
  generateButton.className = "generateButton";
  generateButton.innerText = "Generate 3D";
  prompts_container.appendChild(generateButton);

  // Set up button click event to make an xhr POST request
  generateButton.addEventListener("click", function() {
    // Collect text from all textareas
    var textAreas = prompts_container.querySelectorAll("textarea");
    var promptsData = {};
    
    textAreas.forEach(textArea => {
      promptsData[textArea.id] = textArea.value;
    });

    var xhr = new XMLHttpRequest();
    xhr.open("POST", 'http://localhost:5000/generate3d', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onreadystatechange = function() {
      if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
        console.log("3D Generation Request Successful");
        console.log("Response:", xhr.responseText);
      } else if (xhr.readyState === XMLHttpRequest.DONE) {
        console.log("3D Generation Request Failed");
        console.log("Response:", xhr.responseText);
      }
    };

    xhr.send(JSON.stringify(promptsData));
  });

  //var adjustButton = document.createElement("button");
  //adjustButton.id = "adjust";
  //adjustButton.className = "generateButton";
  //adjustButton.innerText = "Adjust";
  //prompts_container.appendChild(adjustButton);
  // Set up button click event to make an xhr POST request
  /*
  adjustButton.addEventListener("click", function() {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", 'http://localhost:5000/adjust', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onreadystatechange = function() {
      if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
        console.log("3D Generation Request Successful");
        console.log("Response:", xhr.responseText);
      } else if (xhr.readyState === XMLHttpRequest.DONE) {
        console.log("3D Generation Request Failed");
        console.log("Response:", xhr.responseText);
      }
    };

    xhr.send();
  });
  */
  /******************************************************************************************************** */

  // generate layout
  var xhr = new XMLHttpRequest();
  var _ptr = 0;
  var num_iters = 1;
  var n_samples = 6;
  var _tracker = 0;
  xhr.onreadystatechange = function () {
      if (this.status === 200) {
          var data_stream = this.responseText;
          data_stream = data_stream.split('<stop>');
          for (var i = _tracker; i < data_stream.length; i++){
            image_data = data_stream[i];

            // populate
            if(Math.floor(_ptr/num_iters) == 0 && image_data != "" && _ptr < n_samples*num_iters){
              var svg_element = document.getElementById("lgContainer");
              lgContainer.innerHTML = image_data;
              var svg = lgContainer.firstChild;
              scaleSVG(svg, 2.2);
              var progress = document.getElementById("progress");
              progress.innerHTML = "Sample Generated House Layout"
              _ptr++;
              if(Math.floor(_ptr/num_iters) > 0){
                checker_complete(0);
              }
            }
            else if (image_data != "" && _ptr < n_samples*num_iters){
              var smContainer = document.getElementById("sm_img_"+(Math.floor(_ptr/num_iters)-1));
              smContainer.innerHTML = image_data;
              smContainer.onclick = function() {
                miniDisplay = document.getElementById($(this).attr('id'));
                miniDisplaySVG = miniDisplay.firstChild;
                largeDisplay = document.getElementById("lgContainer");
                largeDisplaySVG = largeDisplay.firstChild;
                var s = new XMLSerializer();
                miniDisplay.innerHTML = s.serializeToString(largeDisplaySVG);
                scaleSVG(miniDisplay.firstChild, 1.0/2.2);
                scaleSVG(miniDisplay.firstChild, 0.49);
                largeDisplay.innerHTML = s.serializeToString(miniDisplaySVG);
                scaleSVG(largeDisplay.firstChild, 1.0/0.49);
                scaleSVG(largeDisplay.firstChild, 2.2);
              }
              var svg = smContainer.firstChild;
              scaleSVG(svg, 0.49);
              checker_complete(Math.floor(_ptr/num_iters)-1);
              _ptr++;
            }
          }
          _tracker += (data_stream.length-_tracker)
          if (_tracker == data_stream.length){
            checker_complete(Math.floor(_ptr/num_iters)-1);
          }
      }
  }
  console.log(graph_info);
  xhr.open("POST", 'http://localhost:5000/generate', true);
  xhr.setRequestHeader('Content-Type', 'text/plain');
  xhr.send(JSON.stringify(graph_info));
}

function scaleSVG(svg, factor){
  var svgWidth = parseFloat(svg.getAttributeNS(null, "width"));
  var svgHeight = parseFloat(svg.getAttributeNS(null, "height"));
  // console.log(svgWidth, svgHeight);
  svg.setAttributeNS(null, "width", svgWidth*factor);
  svg.setAttributeNS(null, "height", svgHeight*factor);
  svg.setAttributeNS(null, "viewBox", "0 0 " + svgWidth + " " + svgHeight); 
}

function setTemplate(id){
  data = get_data_object(id);
  add_legend(data);
  draw(data);
  //generate();
}

function get_data_object(template_id) {
  var data = new Object();
  if (template_id==0){
    data.nodes = new vis.DataSet(template_nodes_1);
    data.edges = new vis.DataSet(template_edges_1);
  } 
  else if (template_id==1){
    data.nodes = new vis.DataSet(template_nodes_2);
    data.edges = new vis.DataSet(template_edges_2);
  }
  else if (template_id==2){
    data.nodes = new vis.DataSet(template_nodes_3);
    data.edges = new vis.DataSet(template_edges_3);
  }
  else if (template_id==3){
    data.nodes = new vis.DataSet(template_nodes_4);
    data.edges = new vis.DataSet(template_edges_4);
  }

  return data;
}

window.addEventListener("load", () => {
  var defaultTemplate = 1;
  setTemplate(defaultTemplate);
});
