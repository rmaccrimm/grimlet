import SwiftUI
import Metal
import MetalKit
import CEmuBindings

let SCREEN_W = 240;
let SCREEN_H = 160;

struct Vertex {
    var position: SIMD2<Float>
    var texCoord: SIMD2<Float>
}

class Renderer: NSObject, MTKViewDelegate {
    var emulatorCore: OpaquePointer
    var device: MTLDevice
    var pipeline: MTLRenderPipelineState?
    var vertexBuffer: MTLBuffer?
    var texture: MTLTexture?

    override init() {
        device = MTLCreateSystemDefaultDevice()!
        emulatorCore = core_init()
        super.init()

        let library = try! device.makeDefaultLibrary(bundle: .main)
        
        let t = MTLTextureDescriptor()
        t.pixelFormat = .rgba8Unorm
        t.width = SCREEN_W
        t.height = SCREEN_H
        t.usage = .shaderRead
        texture = device.makeTexture(descriptor: t)

        let p = MTLRenderPipelineDescriptor()
        p.vertexFunction = library.makeFunction(name: "vtx")
        p.fragmentFunction = library.makeFunction(name: "frag")
        p.colorAttachments[0].pixelFormat = t.pixelFormat
        pipeline = try! device.makeRenderPipelineState(descriptor: p)

        let v = [
            Vertex(position: [-1, 1], texCoord: [0, 0]),
            Vertex(position: [-1, -1], texCoord: [0, 1]),
            Vertex(position: [1, 1], texCoord: [1, 0]),
            Vertex(position: [1, -1], texCoord: [1, 1])
        ]
        vertexBuffer = device.makeBuffer(bytes: v, length: v.count * MemoryLayout<Vertex>.stride)
    }
    
    deinit {
        core_drop(emulatorCore)
    }
    
    func draw(in view: MTKView) {
        guard let texture = texture,
              let commandQueue = view.device?.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderPipelineState = pipeline,
              let vertexBuffer = vertexBuffer,
              let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor
        else {
            return
        }
        
        if let pixels = core_get_frame_buffer(emulatorCore) {
            texture.replace(
                region: MTLRegionMake2D(0, 0, SCREEN_W, SCREEN_H),
                mipmapLevel: 0,
                withBytes: UnsafeRawPointer(pixels),
                bytesPerRow: MemoryLayout<Pixel>.size * SCREEN_W
            )
        }

        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderEncoder.setRenderPipelineState(renderPipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.setFragmentTexture(texture, index: 0)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}

struct MetalView: NSViewRepresentable {
    func makeCoordinator() -> Renderer {
        Renderer()
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: context.coordinator.device)
        view.delegate = context.coordinator
        view.preferredFramesPerSecond = 60
        view.colorPixelFormat = .rgba8Unorm
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}

struct ContentView: View {
    var body: some View {
        MetalView()
            .frame(width: CGFloat(SCREEN_W), height: CGFloat(SCREEN_H))
    }
}

@main
struct GrimletApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

