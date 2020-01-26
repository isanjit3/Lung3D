<template id="upload_temp">
    <div>
        <form id="upload_form" role="form" enctype="multipart/form-data" method="POST">
            <input type="file" name="file"  id="file" v-on:change="onFileChange" class="form-control">
            <button  class="btn btn-success btn-block" @click="upload">Upload</button>
        </form>
        <div>{{uploadStatus}}</div>
     </div>
</template>


<script type="text/javascript">
import axios from 'axios';

export default {
    name: "Upload",
    data(){
        return {
            file:null,
            uploadStatus: "Select file to upload"
        }
    },
    methods: {
        onFileChange(e) {
            let files = e.target.files || e.dataTransfer.files;
            if (!files.length)
                return;
            this.createFile(files[0]);
        },
        createFile(file) {
            let reader = new FileReader();
            let vm = this;
            reader.onload = (e) => {
                vm.file = e.target.result;
            };
            reader.readAsDataURL(file);
        },
        upload(){
            const path = 'http://192.168.86.43:5000/upload';
            var data = new FormData();
                data.append('foo', 'bar');
                data.append('file', document.getElementById('file').files[0]);
                axios.post(path, data).then(function (response) {
                    console.log(response)
                    this.uploadStatus = response;
                });
        }
    }

};
   
</script>
