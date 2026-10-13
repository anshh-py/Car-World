$(document).ready(function(){
   $(".card").parent("a").css("text-decoration-line","none"); 
});

(function loadCarWorldAuth() {
   function appendScript(source, onload, onerror) {
      var script = document.createElement("script");
      script.src = source;
      script.onload = onload;
      script.onerror = onerror || function () {};
      document.head.appendChild(script);
   }

   function start() {
      if (!document.getElementById("loginmodal")) return;
      // The homepage loads the shared auth module directly. Avoid a second
      // initialization, which can add duplicate modal controls and listeners.
      if (window.CarWorldAuth) return;

      function loadAuth() {
         appendScript("carworld-auth.js?v=20261012-1");
      }

      if (window.CarWorldAPI) {
         loadAuth();
      } else {
         appendScript("apps/carworld-config.js?v=20261013-2", function () {
            if (!window.CARWORLD_API_BASE_URL) {
               console.error("CarWorld API URL configuration is unavailable; authentication was not initialized.");
               return;
            }
            appendScript("carworld-api.js?v=20261013-1", loadAuth);
         }, function () {
            console.error("Could not load CarWorld API URL configuration; authentication was not initialized.");
         });
      }
   }

   if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", start, { once: true });
   } else {
      start();
   }
})();

(function () {
   function openCatalogSearch(value) {
      var query = value.trim();
      if (!query) return;

      var target = new URL("index.html", window.location.href);
      target.searchParams.set("q", query);
      window.location.href = target.toString();
   }

   document.addEventListener("submit", function (event) {
      if (event.target.id === "car-search-form") return;

      var input = event.target.querySelector('input[type="search"]');
      if (!input || !input.value.trim()) return;
      event.preventDefault();
      openCatalogSearch(input.value);
   });

   document.addEventListener("click", function (event) {
      var button = event.target.closest(".input-group-append button");
      if (!button) return;

      var input = button.closest(".input-group")?.querySelector('input[type="search"]');
      if (!input || input.form) return;
      event.preventDefault();
      openCatalogSearch(input.value);
   });

   document.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" || !event.target.matches('input[type="search"]') || event.target.form) return;
      event.preventDefault();
      openCatalogSearch(event.target.value);
   });
})();
$(function(){

$("#mycollapse").on('shown.bs.collapse',function(){
           $("#showbtn").html('show less brands <b><i class="fas fa-angle-up"></i></b>') 
});
$("#mycollapse").on('hidden.bs.collapse',function(){
           $("#showbtn").html('show more brands <b><i class="fas fa-angle-down"></i></b>') 
});
    $("#signup").click(function(){
       $("#loginmodal").modal('hide');
    });
    $("#login").click(function(){
       $("#loginmodal1").modal('hide');
    });
    
    $("#collapse-1").on('shown.bs.collapse',function(){
           $("#arrows1").html('<i class="fas fa-angle-up close"></i>') 
});
$("#collapse-1").on('hidden.bs.collapse',function(){
           $("#arrows1").html('<b><i class="fas fa-angle-down close"></i></b>') 
});
     $("#collapse-2").on('shown.bs.collapse',function(){
           $("#arrows2").html('<i class="fas fa-angle-up close"></i>') 
});
$("#collapse-2").on('hidden.bs.collapse',function(){
           $("#arrows2").html('<b><i class="fas fa-angle-down close"></i></b>') 
});
     $("#collapse-3").on('shown.bs.collapse',function(){
           $("#arrows3").html('<i class="fas fa-angle-up close"></i>') 
});
$("#collapse-3").on('hidden.bs.collapse',function(){
           $("#arrows3").html('<b><i class="fas fa-angle-down close"></i></b>') 
});
    $("#collapse-4").on('shown.bs.collapse',function(){
           $("#arrows4").html('<i class="fas fa-angle-up close"></i>') 
});
$("#collapse-4").on('hidden.bs.collapse',function(){
           $("#arrows4").html('<b><i class="fas fa-angle-down close"></i></b>') 
});
    $("#collapse-5").on('shown.bs.collapse',function(){
           $("#arrows5").html('<i class="fas fa-angle-up close"></i>') 
});
$("#collapse-5").on('hidden.bs.collapse',function(){
           $("#arrows5").html('<b><i class="fas fa-angle-down close"></i></b>') 
});
    $("#collapse-6").on('shown.bs.collapse',function(){
           $("#arrows6").html('<i class="fas fa-angle-up close"></i>') 
});
$("#collapse-6").on('hidden.bs.collapse',function(){
           $("#arrows6").html('<b><i class="fas fa-angle-down close"></i></b>') 
});
    
    $("#pcollapse").on('shown.bs.collapse',function(){
           $("#readmore").html('Collapse');
});
$("#pcollapse").on('hidden.bs.collapse',function(){
           $("#readmore").html('Read More');
});
});
