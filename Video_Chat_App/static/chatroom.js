var audioMuted = false;
var videoMuted = false;

document.addEventListener("DOMContentLoaded", (event)=>{
    var muteBttn = document.getElementById("bttn_mute");
    var muteVidBttn = document.getElementById("bttn_vid_mute");
    var myVideo = document.getElementById("local_vid");

    muteBttn.addEventListener("click", (event)=>{
        audioMuted = !audioMuted;
        let local_stream = myVideo.srcObject;
        local_stream.getAudioTracks().forEach((track)=>{track.enabled = !audioMuted;});
    });    
    muteVidBttn.addEventListener("click", (event)=>{
        videoMuted = !videoMuted;
        let local_stream = myVideo.srcObject;
        local_stream.getVideoTracks().forEach((track)=>{track.enabled = !videoMuted;});
    });  
    
    startCamera();
    
});


var camera_allowed=false;
var mediaConstraints = {
    audio: true,
    video: {
        height: 360
    } 
};

function startCamera()
{
    navigator.mediaDevices.getUserMedia(mediaConstraints)
    .then((stream)=>{
        document.getElementById("local_vid").srcObject = stream;
        camera_allowed=true;
    })
    .catch((e)=>{
        console.log("Error! Unable to start video! ", e);
        document.getElementById("permission_alert").style.display = "block";
    });
}
