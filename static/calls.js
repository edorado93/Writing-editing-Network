var topicsSelected = [];
var topicsShownToUser = [];
var displayedTopics = [];

function emptyTopics(myNode) {
  while (myNode.firstChild) {
    myNode.removeChild(myNode.firstChild);
  }
}

function getTopics() {

    topicsShownToUser = [];
    topicsSelected = [];

    URL = window.location.href + "getTopics";
    var title = document.getElementById("title").value;
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            data = JSON.parse(this.responseText);
            loadTopics(data["topics"]);
        }
    }
    xmlHttp.open("POST", URL, true);
    xmlHttp.setRequestHeader('Content-Type', 'application/json');
    xmlHttp.send(JSON.stringify({
        "title": title
      }));
}

function selectTopic(e) {
    document.getElementById("alert").setAttribute("hidden", true);
    var element = e.srcElement;
    if (element.classList.contains("active")) {
        element.classList.remove("active");
        topicsSelected.splice(topicsSelected.indexOf(element.innerText), 1);
    } else if (topicsSelected.length < 3){
        element.classList.add("active");
        topicsSelected.push(element.innerText);
    }
}

function animate() {
    setTimeout(function () {
        progress = progress + increment;
        console.log(progress);
        if(progress < maxprogress) {
            $(".progress-bar").attr('aria-valuenow',progress).css('width',progress +'%');
            animate();
        } else {
            document.getElementsByClassName("outer-progress")[0].setAttribute("hidden", true);
        }
    }, timeout);
};


function generateAbstracts(e) {
    if (topicsSelected.length < 3) {
        document.getElementById("alert").removeAttribute("hidden");
    } else {

      document.getElementsByClassName("progress-bar")[0].setAttribute("aria-valuenow", 0);
      document.getElementsByClassName("progress-bar")[0].style = "width: 10%; height: 1em";
      document.getElementsByClassName("outer-abstract")[0].setAttribute("hidden", true);

      var elem = document.getElementsByClassName("outer-progress")[0];
      elem.removeAttribute("hidden");
      progress = 11;      // initial value of your progress bar
      timeout = 10;      // number of milliseconds between each frame
      increment = 0.5;    // increment for each frame
      maxprogress = 500; // when to leave stop running the animation
      animate()

      URL = window.location.href + "getAbstracts";
      var title = document.getElementById("title").value;
      var xmlHttp = new XMLHttpRequest();
      xmlHttp.onreadystatechange = function() {
          if (this.readyState == 4 && this.status == 200) {
              data = JSON.parse(this.responseText);
              maxprogress = 0;
              populateAbstract(data["abstract"]);
          }
      }
      xmlHttp.open("POST", URL, true);
      xmlHttp.setRequestHeader('Content-Type', 'application/json');
      xmlHttp.send(JSON.stringify({
          "title": title,
          "topics": topicsSelected
        }));
    }
}

function populateAbstract(abstract) {
    document.getElementsByClassName("outer-abstract")[0].removeAttribute("hidden");
    document.getElementsByClassName("abstract")[0].innerHTML  = abstract;
}

function loadTopics(topics) {

    document.getElementById("prev-topics").setAttribute("hidden", true);
    displayedTopics = [];
    for(i = 0; i < topics.length; i++) {
        topicsShownToUser.push(topics[i]);
    }
    populateTopics();
}

function prevTopics() {
    if (displayedTopics.length == 76) {
        displayedTopics = displayedTopics.slice(0, displayedTopics.length - 16);
    } else{
        displayedTopics = displayedTopics.slice(0, displayedTopics.length - 20);
    }

    populateTopics();
}

function populateTopics() {
    emptyTopics(document.getElementById("topics-left"));
    emptyTopics(document.getElementById("topics-mid"));
    emptyTopics(document.getElementById("topics-right"));

    var divs = ["topics-mid", "topics-right"];
    var topics_list = document.getElementById("topics-left");
    var j = 0;
    var i = 0;
    for(k = 0; k < topicsShownToUser.length; k++)
    {
        if (displayedTopics.includes(topicsShownToUser[k])) {
            continue;
        }

        displayedTopics.push(topicsShownToUser[k]);
        var entry = document.createElement('li');
        entry.onclick = selectTopic;
        entry.appendChild(document.createTextNode(topicsShownToUser[k]));
        entry.classList.add("list-group-item");

        if (topicsSelected.includes(topicsShownToUser[k])) {
            entry.classList.add("active");
        }

        topics_list.appendChild(entry);
        i++;

        if (i % 4 == 0) {
          topics_list = document.getElementById(divs[j]);
          j++;
        }

        if (i == 10) {
          break;
        }

    }
    document.getElementById("submit").removeAttribute("hidden");
    if (displayedTopics.length > 10) {
        document.getElementById("prev-topics").removeAttribute("hidden");
    } else {
        document.getElementById("prev-topics").setAttribute("hidden", true);
    }

    if (displayedTopics.length == 76) {
        document.getElementById("more-topics").setAttribute("hidden", true);
    } else {
        document.getElementById("more-topics").removeAttribute("hidden");
    }
}
