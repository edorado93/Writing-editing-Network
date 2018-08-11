topicsSelectedCount = 0;

function emptyTopics(myNode) {
  while (myNode.firstChild) {
    myNode.removeChild(myNode.firstChild);
  }
}


function getTopics() {
    emptyTopics(document.getElementById("topics-left"));
    emptyTopics(document.getElementById("topics-mid"));
    emptyTopics(document.getElementById("topics-right"));

    URL = window.location.href + "getTopics";
    var title = document.getElementById("title").value;
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            data = JSON.parse(this.responseText);
            populateTopics(data["topics"]);
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
    element = e.srcElement;
    if (element.classList.contains("active")) {
        element.classList.remove("active");
        topicsSelectedCount--;
    } else if (topicsSelectedCount < 2){
        element.classList.add("active");
        topicsSelectedCount++;
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
    if (topicsSelectedCount < 2) {
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
      var topics = Array.from(document.getElementsByClassName("list-group-item active"))
      var active_topics = [];
      for(i = 0; i < topics.length; i++) {
          active_topics.push(topics[i].innerHTML);
      }
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
          "topics": active_topics
        }));
    }
}

function populateAbstract(abstract) {
    document.getElementsByClassName("outer-abstract")[0].removeAttribute("hidden");
    document.getElementsByClassName("abstract")[0].innerHTML  = abstract;
}


function populateTopics(topics) {
    var divs = ["topics-mid", "topics-right"];
    var topics_list = document.getElementById("topics-left");
    var j = 0;
    for(i = 1; i <= topics.length; i++) {
        var entry = document.createElement('li');
        entry.onclick = selectTopic;
        entry.appendChild(document.createTextNode(topics[i - 1]));
        entry.classList.add("list-group-item");
        topics_list.appendChild(entry);

        if (i % 4 == 0) {
          console.log(divs[j]);
          topics_list = document.getElementById(divs[j]);
          j++;
        }
    }
    document.getElementById("submit").removeAttribute("hidden");
}
