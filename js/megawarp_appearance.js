import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "MegaWarp.appearance",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.category && nodeData.category.startsWith("MegaWarp")) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated
                    ? onNodeCreated.apply(this, arguments)
                    : undefined;
                // Optical flow purple/blue theme
                this.bgcolor = "#0d0a1a";
                this.color = "#1a0f2e";
                return r;
            };
        }
    }
});
