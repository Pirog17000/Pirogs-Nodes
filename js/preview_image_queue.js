import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Store queue state
window.pirogsNodes = window.pirogsNodes || {};

// Add dependencies recursively 
function addNodeDependencies(nodeId, oldOutput, newOutput) {
    if (newOutput[nodeId]) return;
    
    const node = oldOutput[nodeId];
    if (!node) return;
    
    newOutput[nodeId] = node;
    
    // Add all input dependencies
    for (const inputValue of Object.values(node.inputs || {})) {
        if (Array.isArray(inputValue)) {
            addNodeDependencies(inputValue[0], oldOutput, newOutput);
        }
    }
}

// Hook into API to filter workflow
const originalApiQueuePrompt = api.queuePrompt;
api.queuePrompt = async function(index, prompt) {
    if (window.pirogsNodes?.targetNodeId && prompt.output) {
        const oldOutput = prompt.output;
        let newOutput = {};
        
        addNodeDependencies(String(window.pirogsNodes.targetNodeId), oldOutput, newOutput);
        prompt.output = newOutput;
        
        window.pirogsNodes.targetNodeId = null;
    }
    
    return originalApiQueuePrompt.apply(this, arguments);
};

app.registerExtension({
    name: "Pirogs.PreviewImageQueue",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PreviewImageQueue") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                if (originalNodeCreated) {
                    originalNodeCreated.apply(this, arguments);
                }
                
                // Add queue button widget
                this.addWidget("button", "Queue", null, async () => {
                    console.log("Queuing just this node:", this.id);
                    window.pirogsNodes.targetNodeId = this.id;
                    await app.queuePrompt(0);
                });
            };
        }
    }
});