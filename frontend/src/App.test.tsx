import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import App from './App'
import { api } from './api'

vi.mock('./api', () => ({
  api: {
    health: vi.fn(), dashboard: vi.fn(), models: vi.fn(), performance: vi.fn(), system: vi.fn(),
    history: vi.fn(), analyzeImage: vi.fn(), analyzeVideo: vi.fn(), reportUrl: vi.fn(() => '#'),
  },
}))

const model = { id:'resnet50',name:'ResNet50',architecture:'resnet50',dataset:'CIFAKE v3',task:'AI-generated image classification',status:'available',checkpoint:{path:'checkpoint.pt',epoch:16,bytes:100},metrics:{accuracy:.97},test_samples:12000,limitations:'32x32 CIFAKE only'}
const result = {id:'analysis-1',created_at:'2026-11-25T00:00:00Z',media_type:'image' as const,filename:'sample.png',sha256:'a'.repeat(64),metadata:{width:32,height:32,preprocessing:{model_input:'32x32'}},result:{predicted_label:'FAKE',real_probability:.01,fake_probability:.99,confidence_score:.99,confidence_calibrated:false,model_identifier:'resnet50:run:epoch-16',inference_duration_ms:20,device:'cuda'},status:'complete'}

afterEach(cleanup)

beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(api.health).mockResolvedValue({status:'ok',service:'DeepShield AI',version:'1.0.0'})
  vi.mocked(api.models).mockResolvedValue({items:[model]} as any)
})

const mount = (path:string) => render(<MemoryRouter initialEntries={[path]}><App/></MemoryRouter>)

describe('DeepShield workspace', () => {
  it('renders navigation and live dashboard empty state', async () => {
    vi.mocked(api.dashboard).mockResolvedValue({counts:{image:0,video:0},recent:[],models:[model],system:{cuda_available:true,gpu_name:'Test GPU'}} as any)
    mount('/')
    expect(screen.getByText('LOCAL RESEARCH WORKSPACE')).toBeInTheDocument()
    expect(screen.getByRole('link',{name:/Image analysis/})).toBeInTheDocument()
    expect(await screen.findByText('Forensics dashboard')).toBeInTheDocument()
    expect(await screen.findByText('No analyses yet')).toBeInTheDocument()
    expect(await screen.findByText('Backend connected')).toBeInTheDocument()
  })

  it('uploads an image, shows loading, then presents the model result', async () => {
    let finish!: (value:typeof result)=>void
    vi.mocked(api.analyzeImage).mockReturnValue(new Promise(resolve=>{finish=resolve}) as any)
    const {container}=mount('/images')
    await screen.findByText('ResNet50')
    const input=container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input,{target:{files:[new File(['image'],'sample.png',{type:'image/png'})]}})
    expect(screen.getByAltText('Selected image preview')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button',{name:/Run image analysis/}))
    expect(await screen.findByText('Running local inference')).toBeInTheDocument()
    finish(result)
    expect(await screen.findByText('AI-generated indicators')).toBeInTheDocument()
    expect(api.analyzeImage).toHaveBeenCalledOnce()
  })

  it('shows image API errors and saved history records', async () => {
    vi.mocked(api.analyzeImage).mockRejectedValue(new Error('Checkpoint unavailable'))
    const {container,unmount}=mount('/images')
    await screen.findByText('ResNet50')
    fireEvent.change(container.querySelector('input[type="file"]') as HTMLInputElement,{target:{files:[new File(['image'],'bad.png',{type:'image/png'})]}})
    fireEvent.click(screen.getByRole('button',{name:/Run image analysis/}))
    expect(await screen.findByText('Checkpoint unavailable')).toBeInTheDocument()
    unmount()
    vi.mocked(api.history).mockResolvedValue({items:[result],total:1} as any)
    mount('/history')
    expect(await screen.findByText('sample.png')).toBeInTheDocument()
    expect(screen.getByText('analysis-1')).toBeInTheDocument()
    expect(screen.getByLabelText('Filter media type')).toBeInTheDocument()
    expect(screen.getByLabelText('From date')).toBeInTheDocument()
  })

  it('renders the responsive app shell on system health', async () => {
    vi.mocked(api.system).mockResolvedValue({python_version:'3.12.4',platform:'Windows',pytorch_version:'2.15',torchvision_version:'0.29',cuda_available:true,cuda_version:'13.2',gpu_name:'RTX 5070 Laptop GPU',gpu_memory_total_bytes:8e9,ram_total_bytes:24e9,ram_available_bytes:12e9,disk:{project_root_free_bytes:1e9,total_bytes:2e9},video_prediction_status:'unavailable'} as any)
    const {container}=mount('/system')
    expect(await screen.findByText('RTX 5070 Laptop GPU')).toBeInTheDocument()
    expect(container.querySelector('.shell')).toBeInTheDocument()
    expect(container.querySelector('.mobile-menu')).toBeInTheDocument()
  })
})